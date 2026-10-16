"""
SSAE-METAFormer Model Inference
End-to-end PyTorch model for ASD classification using multi-atlas fMRI features
"""

import numpy as np
import torch
import torch.nn as nn
import json
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


def _unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _find_state_key(state_dict: dict, suffix: str) -> str:
    matches = [key for key in state_dict.keys() if key.endswith(suffix)]
    if not matches:
        raise KeyError(f"Could not find checkpoint tensor ending with '{suffix}'")
    return sorted(matches)[0]


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _infer_cfg_from_state_dict(state_dict: dict) -> dict:
    ssae_weight = state_dict[_find_state_key(state_dict, "ssae.encoders.AAL.encoder.0.weight")]
    hidden1 = int(ssae_weight.shape[0])
    input_dim = int(ssae_weight.shape[1])

    hidden2_key = _find_state_key(state_dict, "ssae.encoders.AAL.encoder.4.weight")
    hidden2 = int(state_dict[hidden2_key].shape[0])

    latent_key = _find_state_key(state_dict, "ssae.encoders.AAL.encoder.8.weight")
    latent_dim = int(state_dict[latent_key].shape[0])

    embed_key = _find_state_key(state_dict, "metaformer.embed.weight")
    embed_dim = int(state_dict[embed_key].shape[0])

    fd_embed_key = None
    try:
        fd_embed_key = _find_state_key(state_dict, "metaformer.fd_embed.0.weight")
    except KeyError:
        pass
    fd_embed_dim = int(state_dict[fd_embed_key].shape[0]) if fd_embed_key else 0

    block_suffix = ".attn.in_proj_weight"
    depth = sum(1 for key in state_dict.keys() if key.endswith(block_suffix))

    mlp_key = _find_state_key(state_dict, "metaformer.blocks.0.mlp.0.weight")
    mlp_hidden = int(state_dict[mlp_key].shape[0])

    cfg = {
        "INPUT_DIM": input_dim,
        "HIDDEN_DIM_1": hidden1,
        "HIDDEN_DIM_2": hidden2,
        "LATENT_DIM": latent_dim,
        "NUM_ATLASES": 3,
        "EMBED_DIM": embed_dim,
        "MF_DEPTH": max(depth, 1),
        "MF_HEADS": 8,
        "MF_MLP_RATIO": float(mlp_hidden / embed_dim) if embed_dim else 4.0,
        "DROPOUT": 0.25,
        "USE_MEAN_FD": fd_embed_dim > 0,
        "FD_EMBED_DIM": fd_embed_dim,
    }
    return cfg


# ============================================================================
# Model Architecture Classes (mirrors training)
# ============================================================================

class SSAEEncoder(nn.Module):
    """SSAE Encoder with improved architecture."""
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.encoder(x)


class SSAEDecoder(nn.Module):
    """SSAE Decoder."""
    def __init__(self, latent_dim, hidden_dim_2, hidden_dim_1, output_dim, dropout):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, output_dim),
        )
    
    def forward(self, latent):
        return self.decoder(latent)


class MultiAtlasSSAE(nn.Module):
    """Multi-atlas SSAE: Separate encoder/decoder per atlas."""
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim, dropout):
        super().__init__()
        self.atlas_names = ['AAL', 'Schaefer', 'HarvardOxford']
        
        self.encoders = nn.ModuleDict({
            name: SSAEEncoder(input_dim, hidden_dim_1, hidden_dim_2, latent_dim, dropout)
            for name in self.atlas_names
        })
        
        self.decoders = nn.ModuleDict({
            name: SSAEDecoder(latent_dim, hidden_dim_2, hidden_dim_1, input_dim, dropout)
            for name in self.atlas_names
        })
    
    def encode(self, x):
        """x: (B, 3, input_dim)"""
        latents = []
        for i, name in enumerate(self.atlas_names):
            z = self.encoders[name](x[:, i, :])
            latents.append(z)
        return torch.stack(latents, dim=1)  # (B, 3, latent_dim)


class MetaFormerBlock(nn.Module):
    """MetaFormer block with multi-head self-attention."""
    def __init__(self, dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class METAFormerClassifier(nn.Module):
    """METAFormer classifier for fusing multi-atlas latents."""
    def __init__(self, num_atlases, latent_dim, embed_dim, depth, num_heads, mlp_ratio, dropout, use_mean_fd=False, fd_embed_dim=16):
        super().__init__()
        self.use_mean_fd = use_mean_fd
        self.embed = nn.Linear(latent_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_atlases, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.use_mean_fd:
            self.fd_embed = nn.Sequential(
                nn.Linear(1, fd_embed_dim),
                nn.ReLU(),
            )
            classifier_input_dim = embed_dim + fd_embed_dim
        else:
            self.fd_embed = None
            classifier_input_dim = embed_dim
        
        self.blocks = nn.Sequential(
            *[MetaFormerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, latents, mean_fd=None):
        x = self.embed(latents)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Atlas fusion via mean pooling
        if self.use_mean_fd:
            if mean_fd is None:
                mean_fd = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            elif mean_fd.ndim == 1:
                mean_fd = mean_fd.unsqueeze(-1)
            x = torch.cat([x, self.fd_embed(mean_fd)], dim=-1)
        return self.classifier(x).squeeze(-1)


class SSAEMETAFormer(nn.Module):
    """Complete SSAE-METAFormer model."""
    def __init__(self, cfg):
        super().__init__()
        self.ssae = MultiAtlasSSAE(
            cfg['INPUT_DIM'], cfg['HIDDEN_DIM_1'], cfg['HIDDEN_DIM_2'], 
            cfg['LATENT_DIM'], cfg['DROPOUT']
        )
        self.metaformer = METAFormerClassifier(
            cfg['NUM_ATLASES'], cfg['LATENT_DIM'], cfg['EMBED_DIM'],
            cfg['MF_DEPTH'], cfg['MF_HEADS'], cfg['MF_MLP_RATIO'], cfg['DROPOUT'],
            use_mean_fd=cfg.get('USE_MEAN_FD', False),
            fd_embed_dim=cfg.get('FD_EMBED_DIM', 16),
        )
    
    def forward(self, x, mean_fd=None):
        latents = self.ssae.encode(x)
        logits = self.metaformer(latents, mean_fd=mean_fd)
        return logits
    
    def get_latents(self, x):
        return self.ssae.encode(x)


# ============================================================================
# Inference Wrapper
# ============================================================================

class SSAEMETAFormerInference:
    """Inference wrapper for SSAE-METAFormer model"""
    
    def __init__(self, model_path: str, scaler_dir: str, classification_threshold: float = 0.5):
        """
        Load SSAE-METAFormer model and scalers
        
        Args:
            model_path: Path to ssae_metaformer_optimized.pt
            scaler_dir: Directory containing scaler_*.pkl files
        """
        print("\n[INFO] Loading SSAE-METAFormer model...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.scaler_dir = Path(scaler_dir)
        self.manifest_path = self.scaler_dir / "manifest.json"
        self.classification_threshold = float(classification_threshold)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = _unwrap_checkpoint(checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported checkpoint format in {self.model_path}")

        self.fd_scaler = checkpoint.get("fd_scaler") if isinstance(checkpoint, dict) else None
        self.feature_means = checkpoint.get("feature_means") if isinstance(checkpoint, dict) else None

        manifest = _load_manifest(self.manifest_path)
        cfg = dict(manifest.get("cfg", {})) if isinstance(manifest.get("cfg", {}), dict) else {}
        if not cfg:
            cfg = _infer_cfg_from_state_dict(state_dict)

        cfg.setdefault("NUM_ATLASES", 3)
        cfg.setdefault("MF_HEADS", 1)
        cfg.setdefault("MF_MLP_RATIO", 4.0)
        cfg.setdefault("DROPOUT", 0.25)
        if cfg.get("USE_MEAN_FD") and "FD_EMBED_DIM" not in cfg:
            try:
                cfg["FD_EMBED_DIM"] = int(_find_state_key(state_dict, "metaformer.fd_embed.0.weight") and state_dict[_find_state_key(state_dict, "metaformer.fd_embed.0.weight")].shape[0])
            except Exception:
                cfg["FD_EMBED_DIM"] = 16
        cfg["INPUT_DIM"] = int(cfg["INPUT_DIM"])
        cfg["HIDDEN_DIM_1"] = int(cfg["HIDDEN_DIM_1"])
        cfg["HIDDEN_DIM_2"] = int(cfg["HIDDEN_DIM_2"])
        cfg["LATENT_DIM"] = int(cfg["LATENT_DIM"])
        cfg["NUM_ATLASES"] = int(cfg["NUM_ATLASES"])
        cfg["EMBED_DIM"] = int(cfg["EMBED_DIM"])
        cfg["MF_DEPTH"] = int(cfg["MF_DEPTH"])
        cfg["MF_HEADS"] = int(cfg["MF_HEADS"])
        cfg["MF_MLP_RATIO"] = float(cfg["MF_MLP_RATIO"])
        cfg["DROPOUT"] = float(cfg["DROPOUT"])
        if cfg.get("FD_EMBED_DIM") is not None:
            cfg["FD_EMBED_DIM"] = int(cfg["FD_EMBED_DIM"])
        self.cfg = cfg
        
        # Load model
        try:
            self.model = SSAEMETAFormer(self.cfg).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"  [OK] Model loaded: {self.model_path.name}")
            print(f"  [OK] Model config: {self.cfg}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
        
        # Load scalers
        self.scalers = {}
        self.atlas_names = ['AAL', 'Schaefer', 'HarvardOxford']
        
        for atlas in self.atlas_names:
            scaler_path = self.scaler_dir / f'scaler_{atlas}.pkl'
            try:
                self.scalers[atlas] = joblib.load(scaler_path)
                print(f"  [OK] Scaler loaded: scaler_{atlas}.pkl")
            except Exception as e:
                raise RuntimeError(f"Failed to load scaler from {scaler_path}: {e}")
        
        print("[OK] SSAE-METAFormer ready for inference\n")
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess multi-atlas features using saved scalers
        
        Args:
            features: (3, 50) array - [AAL, Schaefer, HarvardOxford] × 50 features
            
        Returns:
            scaled_features: (3, 50) scaled features
            
        Raises:
            ValueError: If features shape is incorrect
        """
        expected_shape = (self.cfg.get('NUM_ATLASES', 3), self.cfg.get('INPUT_DIM', 50))
        if features.shape != expected_shape:
            raise ValueError(
                f"Expected features shape {expected_shape}, got {features.shape}. "
                f"Features must be {expected_shape[0]} atlases × {expected_shape[1]} connectivity features."
            )
        
        scaled = features.copy().astype(np.float32)
        
        try:
            for i, atlas in enumerate(self.atlas_names):
                # Shape for sklearn: (1, 50) -> transform -> (1, 50)
                scaled[i, :] = self.scalers[atlas].transform(features[i:i+1, :])
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")
        
        return scaled
    
    def predict(self, features: np.ndarray, mean_fd: float | None = None) -> Dict:
        """
        Predict ASD classification from multi-atlas features
        
        Args:
            features: (3, 50) array of multi-atlas connectivity features
            
        Returns:
            {
                'pred_label': 0 (Control) or 1 (ASD),
                'pred_prob': float (0-1) - confidence in ASD class,
                'confidence': float (0-1) - distance from decision boundary,
                'latents': array (3, 32) - latent representation,
                'model': str - model identifier
            }
        """
        try:
            # Preprocess
            features_scaled = self.preprocess(features)
            
            # Prepare tensor: (3, 50) -> (1, 3, 50)
            X_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            mean_fd_value = 0.0 if mean_fd is None else float(mean_fd)
            if self.cfg.get("USE_MEAN_FD") and self.fd_scaler is not None:
                mean_fd_value = float(self.fd_scaler.transform([[mean_fd_value]])[0][0])
            mean_fd_tensor = torch.tensor([[mean_fd_value]], dtype=torch.float32).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(X_tensor, mean_fd_tensor)  # (1,)
                probs = torch.sigmoid(logits)  # (1,)
                latents = self.model.get_latents(X_tensor)  # (1, 3, 32)
            
            pred_prob = float(probs.cpu().numpy()[0])
            pred_label = 1 if pred_prob >= self.classification_threshold else 0
            
            # Confidence: distance from the configured decision boundary.
            confidence = abs(pred_prob - self.classification_threshold) * 2  # [0, 1]
            confidence = float(min(max(confidence, 0.0), 1.0))
            
            return {
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'confidence': float(confidence),
                'classification_threshold': float(self.classification_threshold),
                'latents': latents.cpu().numpy()[0],  # (3, 32)
                'model': 'SSAE-METAFormer'
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def batch_predict(self, features_batch: np.ndarray) -> List[Dict]:
        """
        Batch prediction - more efficient than individual calls
        
        Args:
            features_batch: (N, 3, 50) array
            
        Returns:
            List of N prediction dicts
        """
        predictions = []
        for features in features_batch:
            pred = self.predict(features)
            predictions.append(pred)
        return predictions
    
    def get_device_info(self) -> Dict[str, str]:
        """Get device information for logging/debugging"""
        return {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__
        }


def get_severity_bucket(pred_prob: float) -> str:
    """
    Map prediction probability to severity bucket
    
    Args:
        pred_prob: Probability from model [0, 1]
        
    Returns:
        'low', 'medium', or 'high'
    """
    if pred_prob < 0.33:
        return 'low'
    elif pred_prob < 0.67:
        return 'medium'
    else:
        return 'high'


def get_confidence_level(confidence: float) -> str:
    """
    Map confidence to confidence level
    
    Args:
        confidence: Confidence score [0, 1]
        
    Returns:
        'low', 'moderate', or 'high'
    """
    if confidence < 0.33:
        return 'low'
    elif confidence < 0.67:
        return 'moderate'
    else:
        return 'high'
