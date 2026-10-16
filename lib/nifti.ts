import type { NVImage } from "@niivue/niivue";

export type NiftiMetadata = {
  dimensions: number[];
  voxelSpacing: number[];
  affine: number[][];
  orientation: string;
  datatype: string;
  datatypeCode: number;
  intensityRange: [number, number];
};

const datatypeNames: Record<number, string> = {
  0: "Unknown",
  1: "Binary",
  2: "UINT8",
  4: "INT16",
  8: "INT32",
  16: "FLOAT32",
  32: "COMPLEX64",
  64: "FLOAT64",
  128: "RGB24",
  256: "INT8",
  512: "UINT16",
  768: "UINT32",
  1024: "INT64",
  1280: "UINT64",
  1792: "COMPLEX128",
  2304: "RGBA32",
};

function getAxisCode(column: number[]) {
  const axis = column.reduce(
    (best, value, index) => (Math.abs(value) > Math.abs(column[best]) ? index : best),
    0,
  );
  const positive = column[axis] >= 0;
  if (axis === 0) return positive ? "R" : "L";
  if (axis === 1) return positive ? "A" : "P";
  return positive ? "S" : "I";
}

function inferOrientation(affine: number[][]) {
  if (affine.length < 3 || affine.some((row) => row.length < 3)) return "Не определена";
  return [0, 1, 2]
    .map((columnIndex) => getAxisCode([affine[0][columnIndex], affine[1][columnIndex], affine[2][columnIndex]]))
    .join("");
}

export function extractNiftiMetadata(volume: NVImage): NiftiMetadata {
  const header = volume.hdr as unknown as {
    dims?: number[];
    pixDims?: number[];
    affine?: number[][];
    datatypeCode?: number;
  };
  const rawDims = volume.dims ?? header?.dims ?? [];
  const rawPixDims = volume.pixDims ?? header?.pixDims ?? [];
  const dimensions = rawDims.length >= 4 ? rawDims.slice(1, 4) : rawDims.slice(0, 3);
  const voxelSpacing = rawPixDims.length >= 4 ? rawPixDims.slice(1, 4) : rawPixDims.slice(0, 3);
  const affine = (volume.originalAffine ?? header?.affine ?? []).map((row) => Array.from(row));
  const datatypeCode = Number(header?.datatypeCode ?? 0);
  const min = Number(volume.robust_min ?? volume.global_min ?? volume.cal_min ?? 0);
  const max = Number(volume.robust_max ?? volume.global_max ?? volume.cal_max ?? 1);

  return {
    dimensions: dimensions.map(Number),
    voxelSpacing: voxelSpacing.map((value) => Math.abs(Number(value))),
    affine,
    orientation: inferOrientation(affine),
    datatype: datatypeNames[datatypeCode] ?? `NIfTI code ${datatypeCode}`,
    datatypeCode,
    intensityRange: [Number.isFinite(min) ? min : 0, Number.isFinite(max) && max !== min ? max : 1],
  };
}
