"use client";

import {
  AlertCircle,
  Box,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Columns3,
  Crosshair,
  Database,
  FileImage,
  Files,
  HardDrive,
  Info,
  Layers3,
  LoaderCircle,
  Moon,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  Sun,
  TableProperties,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { useTheme } from "next-themes";
import Papa from "papaparse";
import * as React from "react";
import { Niivue, NVImage, SHOW_RENDER, SLICE_TYPE } from "@niivue/niivue";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn, formatBytes } from "@/lib/utils";
import { extractNiftiMetadata, type NiftiMetadata } from "@/lib/nifti";
import {
  getDiagnosis,
  getSex,
  inferSubjectId,
  normalizeRecord,
  normalizeSubjectId,
  type PhenotypicRecord,
} from "@/lib/phenotypic";

type ViewerMode = "axial" | "sagittal" | "coronal" | "multiplanar" | "render";

type MRIEntry = {
  id: string;
  file: File;
  name: string;
  size: number;
  subjectId: string;
};

type SlicePosition = { x: number; y: number; z: number };

const modeOptions: { value: ViewerMode; label: string; shortLabel: string; icon: React.ElementType }[] = [
  { value: "axial", label: "Аксиальный 2D", shortLabel: "Axial", icon: Layers3 },
  { value: "sagittal", label: "Сагиттальный 2D", shortLabel: "Sagittal", icon: Columns3 },
  { value: "coronal", label: "Корональный 2D", shortLabel: "Coronal", icon: ScanLine },
  { value: "multiplanar", label: "Три плоскости", shortLabel: "MPR", icon: Crosshair },
  { value: "render", label: "Объёмный рендеринг", shortLabel: "3D", icon: Box },
];

const colormapOptions = [
  { value: "gray", label: "Gray" },
  { value: "bone", label: "Bone" },
  { value: "hot", label: "Hot" },
  { value: "viridis", label: "Viridis" },
  { value: "plasma", label: "Plasma" },
  { value: "winter", label: "Winter" },
];

const modeToSliceType: Record<ViewerMode, SLICE_TYPE> = {
  axial: SLICE_TYPE.AXIAL,
  sagittal: SLICE_TYPE.SAGITTAL,
  coronal: SLICE_TYPE.CORONAL,
  multiplanar: SLICE_TYPE.MULTIPLANAR,
  render: SLICE_TYPE.RENDER,
};

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => setMounted(true), []);

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={resolvedTheme === "dark" ? "Включить светлую тему" : "Включить тёмную тему"}
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      {mounted && resolvedTheme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}

function Panel({ className, children }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <section className={cn("rounded-xl border bg-card text-card-foreground shadow-panel", className)}>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value?: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-border/70 py-2.5 last:border-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="max-w-[62%] break-words text-right text-xs font-medium">{value || "—"}</dd>
    </div>
  );
}

function EmptyMetadata() {
  return (
    <div className="flex flex-col items-center px-5 py-10 text-center">
      <div className="grid h-10 w-10 place-items-center rounded-lg border bg-muted">
        <TableProperties className="h-4 w-4 text-muted-foreground" />
      </div>
      <p className="mt-3 text-sm font-medium">Нет данных заголовка</p>
      <p className="mt-1 text-xs leading-5 text-muted-foreground">Выберите NIfTI-файл, чтобы прочитать метаданные.</p>
    </div>
  );
}

export default function MRIViewer() {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const niivueRef = React.useRef<Niivue | null>(null);
  const volumeRef = React.useRef<NVImage | null>(null);
  const mriInputRef = React.useRef<HTMLInputElement>(null);
  const csvInputRef = React.useRef<HTMLInputElement>(null);
  const loadTokenRef = React.useRef(0);
  const { resolvedTheme } = useTheme();

  const [entries, setEntries] = React.useState<MRIEntry[]>([]);
  const [activeId, setActiveId] = React.useState<string | null>(null);
  const [mode, setMode] = React.useState<ViewerMode>("multiplanar");
  const [metadata, setMetadata] = React.useState<NiftiMetadata | null>(null);
  const [viewerReady, setViewerReady] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [dragActive, setDragActive] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [brightness, setBrightness] = React.useState(0);
  const [contrast, setContrast] = React.useState(1);
  const [colormap, setColormap] = React.useState("gray");
  const [slicePosition, setSlicePosition] = React.useState<SlicePosition>({ x: 0, y: 0, z: 0 });
  const [sliderDimensions, setSliderDimensions] = React.useState([1, 1, 1]);
  const [csvName, setCsvName] = React.useState<string | null>(null);
  const [csvRowCount, setCsvRowCount] = React.useState(0);
  const [phenotypicIndex, setPhenotypicIndex] = React.useState<Map<string, PhenotypicRecord>>(new Map());

  const activeEntry = React.useMemo(
    () => entries.find((entry) => entry.id === activeId) ?? null,
    [activeId, entries],
  );
  const activePhenotype = React.useMemo(
    () => phenotypicIndex.get(normalizeSubjectId(activeEntry?.subjectId)),
    [activeEntry?.subjectId, phenotypicIndex],
  );
  const diagnosis = getDiagnosis(activePhenotype);

  React.useEffect(() => {
    let cancelled = false;

    async function initializeViewer() {
      if (!canvasRef.current) return;
      try {
        const nv = new Niivue({
          dragAndDropEnabled: false,
          show3Dcrosshair: true,
          isOrientCube: true,
          isColorbar: false,
          crosshairColor: [0.16, 0.85, 0.75, 0.9],
          crosshairWidth: 1,
          textHeight: 0.03,
          multiplanarPadPixels: 4,
        });
        await nv.attachToCanvas(canvasRef.current);
        if (cancelled) {
          nv.cleanup();
          return;
        }
        nv.setMultiplanarLayout(2);
        nv.opts.multiplanarShowRender = SHOW_RENDER.NEVER;
        nv.setSliceType(SLICE_TYPE.MULTIPLANAR);

        nv.addEventListener("locationChange", () => {
          const dims = sliderDimensionsRef.current;
          const position = Array.from(nv.scene.crosshairPos ?? [0.5, 0.5, 0.5]);
          setSlicePosition({
            x: Math.round(position[0] * Math.max(0, dims[0] - 1)),
            y: Math.round(position[1] * Math.max(0, dims[1] - 1)),
            z: Math.round(position[2] * Math.max(0, dims[2] - 1)),
          });
        });

        niivueRef.current = nv;
        setViewerReady(true);
      } catch (cause) {
        setError(
          cause instanceof Error
            ? `Не удалось запустить WebGL: ${cause.message}`
            : "Не удалось запустить WebGL2. Проверьте поддержку WebGL в браузере.",
        );
      }
    }

    void initializeViewer();

    return () => {
      cancelled = true;
      loadTokenRef.current += 1;
      const nv = niivueRef.current;
      if (nv) {
        for (const volume of [...nv.volumes]) nv.removeVolume(volume);
        nv.cleanup();
      }
      volumeRef.current = null;
      niivueRef.current = null;
    };
  }, []);

  const sliderDimensionsRef = React.useRef(sliderDimensions);
  React.useEffect(() => {
    sliderDimensionsRef.current = sliderDimensions;
  }, [sliderDimensions]);

  React.useEffect(() => {
    const nv = niivueRef.current;
    if (!nv) return;
    const isDark = resolvedTheme !== "light";
    nv.opts.backColor = isDark ? [0.025, 0.04, 0.07, 1] : [0.94, 0.96, 0.97, 1];
    nv.drawScene();
  }, [resolvedTheme, viewerReady]);

  const applyAppearance = React.useCallback(
    (nv: Niivue, volume: NVImage) => {
      const min = metadata?.intensityRange[0] ?? Number(volume.robust_min ?? volume.global_min ?? 0);
      const max = metadata?.intensityRange[1] ?? Number(volume.robust_max ?? volume.global_max ?? 1);
      const fullWidth = Math.max(Number.EPSILON, max - min);
      const center = (min + max) / 2 + (brightness / 100) * fullWidth * 0.5;
      const windowWidth = fullWidth / Math.max(0.25, contrast);
      volume.cal_min = center - windowWidth / 2;
      volume.cal_max = center + windowWidth / 2;
      nv.setColormap(volume.id, colormap);
      nv.updateGLVolume();
    },
    [brightness, colormap, contrast, metadata?.intensityRange],
  );

  React.useEffect(() => {
    const nv = niivueRef.current;
    const volume = volumeRef.current;
    if (nv && volume) applyAppearance(nv, volume);
  }, [applyAppearance]);

  React.useEffect(() => {
    const nv = niivueRef.current;
    if (!viewerReady || !nv) return;
    const token = ++loadTokenRef.current;

    for (const volume of [...nv.volumes]) nv.removeVolume(volume);
    volumeRef.current = null;
    setMetadata(null);

    if (!activeEntry) {
      setSliderDimensions([1, 1, 1]);
      setSlicePosition({ x: 0, y: 0, z: 0 });
      nv.drawScene();
      return;
    }

    const currentEntry = activeEntry;
    const currentNv = nv;

    async function loadVolume() {
      setIsLoading(true);
      setError(null);
      try {
        const volume = await NVImage.loadFromFile({ file: currentEntry.file });
        if (token !== loadTokenRef.current) return;
        currentNv.addVolume(volume);
        volumeRef.current = volume;
        const nextMetadata = extractNiftiMetadata(volume);
        const displayDimsRaw = volume.dimsRAS ?? volume.dims ?? nextMetadata.dimensions;
        const displayDims = (displayDimsRaw.length >= 4 ? displayDimsRaw.slice(1, 4) : displayDimsRaw.slice(0, 3)).map(
          (value) => Math.max(1, Number(value) || 1),
        );
        setMetadata(nextMetadata);
        setSliderDimensions(displayDims);
        setSlicePosition({
          x: Math.floor((displayDims[0] - 1) / 2),
          y: Math.floor((displayDims[1] - 1) / 2),
          z: Math.floor((displayDims[2] - 1) / 2),
        });
        currentNv.scene.crosshairPos[0] = 0.5;
        currentNv.scene.crosshairPos[1] = 0.5;
        currentNv.scene.crosshairPos[2] = 0.5;
        currentNv.setSliceType(modeToSliceType[mode]);
        if (mode === "multiplanar") {
          currentNv.setMultiplanarLayout(2);
          currentNv.opts.multiplanarShowRender = SHOW_RENDER.NEVER;
        }
        applyAppearance(currentNv, volume);
        currentNv.drawScene();
      } catch (cause) {
        if (token !== loadTokenRef.current) return;
        setError(
          cause instanceof Error
            ? `Не удалось открыть «${currentEntry.name}»: ${cause.message}`
            : `Не удалось открыть «${currentEntry.name}». Проверьте NIfTI-файл.`,
        );
      } finally {
        if (token === loadTokenRef.current) setIsLoading(false);
      }
    }

    void loadVolume();
  }, [activeEntry?.id, viewerReady]);

  function addMRIFiles(fileList: FileList | File[]) {
    const incoming = Array.from(fileList);
    const valid: MRIEntry[] = [];
    const rejected: string[] = [];

    for (const file of incoming) {
      const lowerName = file.name.toLowerCase();
      if ((!lowerName.endsWith(".nii") && !lowerName.endsWith(".nii.gz")) || file.size === 0) {
        rejected.push(file.name);
        continue;
      }
      const duplicate = entries.some(
        (entry) => entry.name === file.name && entry.size === file.size && entry.file.lastModified === file.lastModified,
      );
      if (duplicate) continue;
      valid.push({
        id: crypto.randomUUID(),
        file,
        name: file.name,
        size: file.size,
        subjectId: inferSubjectId(file.name),
      });
    }

    if (rejected.length) {
      setError(`Пропущены неподдерживаемые или пустые файлы: ${rejected.join(", ")}`);
    } else if (valid.length) {
      setError(null);
    }
    if (!valid.length) return;
    setEntries((current) => [...current, ...valid]);
    setActiveId((current) => current ?? valid[0].id);
  }

  function removeEntry(id: string) {
    setEntries((current) => {
      const index = current.findIndex((entry) => entry.id === id);
      const next = current.filter((entry) => entry.id !== id);
      if (activeId === id) setActiveId(next[Math.min(index, next.length - 1)]?.id ?? null);
      return next;
    });
  }

  function changeMode(nextMode: ViewerMode) {
    setMode(nextMode);
    const nv = niivueRef.current;
    if (!nv) return;
    nv.setSliceType(modeToSliceType[nextMode]);
    if (nextMode === "multiplanar") {
      nv.setMultiplanarLayout(2);
      nv.opts.multiplanarShowRender = SHOW_RENDER.NEVER;
    }
    nv.drawScene();
  }

  function setSlice(axis: keyof SlicePosition, nextValue: number) {
    const nv = niivueRef.current;
    if (!nv || !volumeRef.current) return;
    const axisIndex = axis === "x" ? 0 : axis === "y" ? 1 : 2;
    const max = Math.max(1, sliderDimensions[axisIndex] - 1);
    nv.scene.crosshairPos[axisIndex] = nextValue / max;
    setSlicePosition((current) => ({ ...current, [axis]: nextValue }));
    nv.drawScene();
  }

  function loadPhenotypicCsv(file?: File) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv") || file.size === 0) {
      setError("Выберите непустой CSV-файл ABIDE I.");
      return;
    }

    Papa.parse<Record<string, unknown>>(file, {
      header: true,
      skipEmptyLines: "greedy",
      transformHeader: (header) => header.replace(/^\uFEFF/, "").trim(),
      complete: (result) => {
        const records = result.data.map(normalizeRecord);
        if (!records.length || !("SUB_ID" in records[0])) {
          setError("В CSV не найден обязательный столбец SUB_ID.");
          return;
        }
        const nextIndex = new Map<string, PhenotypicRecord>();
        for (const record of records) {
          const id = normalizeSubjectId(record.SUB_ID);
          if (id) nextIndex.set(id, record);
        }
        setPhenotypicIndex(nextIndex);
        setCsvName(file.name);
        setCsvRowCount(nextIndex.size);
        setError(result.errors.length ? `CSV загружен, но содержит ${result.errors.length} предупреждений.` : null);
      },
      error: (parseError) => setError(`Не удалось прочитать CSV: ${parseError.message}`),
    });
  }

  function updateActiveSubjectId(value: string) {
    if (!activeId) return;
    setEntries((current) => current.map((entry) => (entry.id === activeId ? { ...entry, subjectId: value } : entry)));
  }

  const age = activePhenotype?.AGE_AT_SCAN || activePhenotype?.AGE;
  const site = activePhenotype?.SITE_ID || activePhenotype?.SITE;

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b bg-background/92 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1920px] items-center justify-between px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground shadow-sm shadow-primary/20">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-sm font-semibold tracking-tight sm:text-base">ABIDE MRI Viewer</h1>
                <Badge variant="outline" className="hidden font-mono text-[10px] sm:inline-flex">
                  NIfTI · WebGL2
                </Badge>
              </div>
              <p className="hidden text-xs text-muted-foreground sm:block">Структурная нейровизуализация ABIDE I</p>
            </div>
          </div>
          <div className="flex items-center gap-1 sm:gap-3">
            <div className="hidden items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-xs text-muted-foreground md:flex">
              <ShieldCheck className="h-3.5 w-3.5 text-primary" />
              Файлы остаются в браузере
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1920px] p-3 sm:p-4 lg:p-5">
        {error && (
          <div
            className="mb-4 flex items-start justify-between gap-4 rounded-lg border border-destructive/25 bg-destructive/10 px-4 py-3 text-sm text-destructive"
            role="alert"
          >
            <span className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </span>
            <button className="text-xs font-medium underline-offset-4 hover:underline" onClick={() => setError(null)}>
              Закрыть
            </button>
          </div>
        )}

        <div className="grid items-start gap-4 xl:grid-cols-[280px_minmax(0,1fr)_330px]">
          <aside className="grid gap-4 md:grid-cols-2 xl:sticky xl:top-20 xl:grid-cols-1">
            <Panel className="overflow-hidden">
              <div className="flex items-center justify-between border-b px-4 py-3.5">
                <div>
                  <h2 className="text-sm font-semibold">Исследования</h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">{entries.length} загружено</p>
                </div>
                <Files className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="p-3">
                <input
                  ref={mriInputRef}
                  className="sr-only"
                  type="file"
                  accept=".nii,.nii.gz,application/gzip"
                  multiple
                  onChange={(event) => {
                    if (event.target.files) addMRIFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                <button
                  type="button"
                  className={cn(
                    "group flex w-full flex-col items-center rounded-lg border border-dashed px-4 py-6 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    dragActive ? "border-primary bg-primary/10" : "border-border bg-muted/25 hover:border-primary/60 hover:bg-accent/40",
                  )}
                  onClick={() => mriInputRef.current?.click()}
                  onDragEnter={(event) => {
                    event.preventDefault();
                    setDragActive(true);
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={(event) => {
                    event.preventDefault();
                    setDragActive(false);
                  }}
                  onDrop={(event) => {
                    event.preventDefault();
                    setDragActive(false);
                    addMRIFiles(event.dataTransfer.files);
                  }}
                >
                  <span className="grid h-9 w-9 place-items-center rounded-full bg-primary/10 text-primary transition-transform group-hover:-translate-y-0.5">
                    <UploadCloud className="h-4 w-4" />
                  </span>
                  <span className="mt-3 text-sm font-medium">Добавить MRI</span>
                  <span className="mt-1 text-xs leading-5 text-muted-foreground">Перетащите .nii / .nii.gz<br />или выберите файлы</span>
                </button>
              </div>

              {entries.length > 0 && (
                <div className="scrollbar-thin max-h-72 space-y-1 overflow-y-auto border-t p-2">
                  {entries.map((entry) => {
                    const isActive = entry.id === activeId;
                    const record = phenotypicIndex.get(normalizeSubjectId(entry.subjectId));
                    const itemDiagnosis = getDiagnosis(record);
                    return (
                      <div
                        key={entry.id}
                        className={cn(
                          "group flex items-center gap-2 rounded-lg border px-2.5 py-2 transition-colors",
                          isActive ? "border-primary/35 bg-primary/10" : "border-transparent hover:bg-muted/70",
                        )}
                      >
                        <button className="flex min-w-0 flex-1 items-center gap-2.5 text-left" onClick={() => setActiveId(entry.id)}>
                          <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-md", isActive ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>
                            <FileImage className="h-4 w-4" />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-xs font-medium">{entry.name}</span>
                            <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                              {entry.subjectId ? `SUB ${entry.subjectId}` : "SUB_ID не найден"}
                              {itemDiagnosis && <span>· {itemDiagnosis}</span>}
                            </span>
                          </span>
                        </button>
                        <button
                          className="rounded p-1 text-muted-foreground opacity-60 hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
                          aria-label={`Удалить ${entry.name}`}
                          onClick={() => removeEntry(entry.id)}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </Panel>

            <Panel className="overflow-hidden">
              <div className="flex items-center justify-between border-b px-4 py-3.5">
                <div>
                  <h2 className="text-sm font-semibold">Фенотипические данные</h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">Сопоставление по SUB_ID</p>
                </div>
                <Database className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="p-3">
                <input
                  ref={csvInputRef}
                  className="sr-only"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => {
                    loadPhenotypicCsv(event.target.files?.[0]);
                    event.target.value = "";
                  }}
                />
                {csvName ? (
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <div className="flex items-start gap-2.5">
                      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                        <CheckCircle2 className="h-4 w-4" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium">{csvName}</p>
                        <p className="mt-1 text-[11px] text-muted-foreground">{csvRowCount.toLocaleString("ru-RU")} записей индексировано</p>
                      </div>
                    </div>
                    <Button variant="outline" size="sm" className="mt-3 w-full" onClick={() => csvInputRef.current?.click()}>
                      Заменить CSV
                    </Button>
                  </div>
                ) : (
                  <Button variant="outline" className="w-full justify-start" onClick={() => csvInputRef.current?.click()}>
                    <UploadCloud className="h-4 w-4" />
                    Загрузить ABIDE I CSV
                  </Button>
                )}
                <p className="mt-2 text-[10px] leading-4 text-muted-foreground">Ожидается Phenotypic_V1_0b_preprocessed.csv со столбцом SUB_ID.</p>
              </div>
            </Panel>
          </aside>

          <section className="min-w-0 space-y-4">
            <Panel className="overflow-hidden">
              <div className="flex flex-col justify-between gap-3 border-b px-3 py-3 sm:flex-row sm:items-center sm:px-4">
                <div className="flex min-w-0 items-center gap-2">
                  <CircleDot className={cn("h-4 w-4 shrink-0", activeEntry ? "text-primary" : "text-muted-foreground")} />
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium">{activeEntry?.name ?? "MRI не выбран"}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {activeEntry ? `${formatBytes(activeEntry.size)} · локальный файл` : "Добавьте структурный NIfTI-снимок"}
                    </p>
                  </div>
                </div>
                <div className="scrollbar-thin flex max-w-full items-center gap-1 overflow-x-auto rounded-lg bg-muted p-1" aria-label="Режим просмотра">
                  {modeOptions.map((option) => {
                    const Icon = option.icon;
                    return (
                      <button
                        key={option.value}
                        className={cn(
                          "flex h-8 shrink-0 items-center gap-1.5 rounded-md px-2.5 text-[11px] font-medium transition-colors",
                          mode === option.value ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
                        )}
                        title={option.label}
                        onClick={() => changeMode(option.value)}
                      >
                        <Icon className="h-3.5 w-3.5" />
                        {option.shortLabel}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="viewer-grid relative h-[54vh] min-h-[430px] overflow-hidden bg-slate-950 sm:h-[62vh] lg:min-h-[560px] xl:h-[calc(100vh-245px)] xl:max-h-[780px]">
                <canvas ref={canvasRef} className="block h-full w-full touch-none outline-none" aria-label="Niivue MRI canvas" />

                {!activeEntry && !isLoading && (
                  <div className="pointer-events-none absolute inset-0 grid place-items-center bg-slate-950/20 p-6">
                    <div className="max-w-sm text-center text-slate-200">
                      <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-white/10 bg-white/5 backdrop-blur">
                        <BrainCircuit className="h-6 w-6 text-teal-300" />
                      </div>
                      <h2 className="mt-5 text-base font-semibold">Откройте структурный MRI</h2>
                      <p className="mt-2 text-sm leading-6 text-slate-400">Загрузите настоящий файл ABIDE I в формате .nii или .nii.gz. Данные не отправляются на сервер.</p>
                    </div>
                  </div>
                )}

                {isLoading && (
                  <div className="absolute inset-0 grid place-items-center bg-slate-950/70 backdrop-blur-sm" aria-live="polite">
                    <div className="flex flex-col items-center text-slate-100">
                      <LoaderCircle className="h-7 w-7 animate-spin text-teal-300" />
                      <p className="mt-3 text-sm font-medium">Разбор NIfTI-тома…</p>
                      <p className="mt-1 max-w-xs truncate text-xs text-slate-400">{activeEntry?.name}</p>
                    </div>
                  </div>
                )}

                <div className="pointer-events-none absolute bottom-3 left-3 hidden rounded-md border border-white/10 bg-slate-950/70 px-2.5 py-1.5 text-[10px] text-slate-400 backdrop-blur sm:block">
                  Колесо — срез · ЛКМ — курсор · ПКМ — окно интенсивности
                </div>
              </div>
            </Panel>

            <Panel className="p-4">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Crosshair className="h-4 w-4 text-primary" />
                  <h2 className="text-sm font-semibold">Положение среза</h2>
                </div>
                <span className="text-[10px] text-muted-foreground">индекс вокселя</span>
              </div>
              <div className="grid gap-5 md:grid-cols-3">
                {([
                  ["x", "Sagittal", "X"],
                  ["y", "Coronal", "Y"],
                  ["z", "Axial", "Z"],
                ] as const).map(([axis, label, coordinate], index) => (
                  <div key={axis}>
                    <div className="mb-2 flex items-center justify-between text-xs">
                      <span className="font-medium">{label} <span className="text-muted-foreground">({coordinate})</span></span>
                      <span className="font-mono text-muted-foreground">{slicePosition[axis]} / {Math.max(0, sliderDimensions[index] - 1)}</span>
                    </div>
                    <Slider
                      min={0}
                      max={Math.max(0, sliderDimensions[index] - 1)}
                      step={1}
                      value={[slicePosition[axis]]}
                      disabled={!metadata}
                      onValueChange={([value]) => setSlice(axis, value)}
                      aria-label={`${label} slice`}
                    />
                  </div>
                ))}
              </div>
            </Panel>
          </section>

          <aside className="grid gap-4 md:grid-cols-2 xl:sticky xl:top-20 xl:grid-cols-1">
            <Panel className="overflow-hidden">
              <div className="flex items-center justify-between border-b px-4 py-3.5">
                <div>
                  <h2 className="text-sm font-semibold">Участник</h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">ABIDE I phenotype</p>
                </div>
                {diagnosis && (
                  <Badge variant={diagnosis === "ASD" ? "warning" : diagnosis === "Control" ? "success" : "outline"}>{diagnosis}</Badge>
                )}
              </div>
              <div className="px-4 py-3">
                <label className="text-[11px] font-medium text-muted-foreground" htmlFor="subject-id">SUB_ID</label>
                <div className="mt-1.5 flex items-center rounded-md border bg-background px-3 focus-within:ring-2 focus-within:ring-ring">
                  <input
                    id="subject-id"
                    className="h-9 min-w-0 flex-1 bg-transparent text-sm font-medium outline-none placeholder:text-muted-foreground"
                    value={activeEntry?.subjectId ?? ""}
                    disabled={!activeEntry}
                    placeholder="Например, 0051456"
                    inputMode="numeric"
                    onChange={(event) => updateActiveSubjectId(event.target.value)}
                  />
                  {activePhenotype && <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
                </div>
                {activeEntry && csvName && !activePhenotype && (
                  <p className="mt-2 flex items-start gap-1.5 text-[10px] leading-4 text-amber-600 dark:text-amber-400">
                    <Info className="mt-0.5 h-3 w-3 shrink-0" /> Запись с этим SUB_ID в CSV не найдена.
                  </p>
                )}
                <dl className="mt-3">
                  <Field label="Группа" value={diagnosis} />
                  <Field label="Возраст" value={age ? `${age} лет` : undefined} />
                  <Field label="Пол" value={getSex(activePhenotype)} />
                  <Field label="Центр" value={site} />
                </dl>
              </div>
            </Panel>

            <Panel className="overflow-hidden">
              <div className="flex items-center justify-between border-b px-4 py-3.5">
                <div>
                  <h2 className="text-sm font-semibold">Визуализация</h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">Окно интенсивности</p>
                </div>
                <SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
              </div>
              <div className="space-y-5 p-4">
                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <label className="font-medium">Яркость</label>
                    <span className="font-mono text-muted-foreground">{brightness > 0 ? "+" : ""}{brightness}</span>
                  </div>
                  <Slider min={-100} max={100} step={1} value={[brightness]} disabled={!metadata} onValueChange={([value]) => setBrightness(value)} />
                </div>
                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <label className="font-medium">Контраст</label>
                    <span className="font-mono text-muted-foreground">{contrast.toFixed(2)}×</span>
                  </div>
                  <Slider min={0.25} max={3} step={0.05} value={[contrast]} disabled={!metadata} onValueChange={([value]) => setContrast(value)} />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-medium" htmlFor="colormap">Цветовая схема</label>
                  <select
                    id="colormap"
                    className="h-9 w-full rounded-md border bg-background px-3 text-xs outline-none focus:ring-2 focus:ring-ring"
                    value={colormap}
                    disabled={!metadata}
                    onChange={(event) => setColormap(event.target.value)}
                  >
                    {colormapOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  disabled={!metadata}
                  onClick={() => {
                    setBrightness(0);
                    setContrast(1);
                    setColormap("gray");
                  }}
                >
                  Сбросить настройки
                </Button>
              </div>
            </Panel>

            <Panel className="overflow-hidden md:col-span-2 xl:col-span-1">
              <div className="flex items-center justify-between border-b px-4 py-3.5">
                <div>
                  <h2 className="text-sm font-semibold">NIfTI-метаданные</h2>
                  <p className="mt-0.5 text-xs text-muted-foreground">Заголовок активного тома</p>
                </div>
                <HardDrive className="h-4 w-4 text-muted-foreground" />
              </div>
              {!metadata ? (
                <EmptyMetadata />
              ) : (
                <div className="px-4 py-2">
                  <dl>
                    <Field label="Dimensions" value={metadata.dimensions.join(" × ")} />
                    <Field label="Voxel spacing" value={`${metadata.voxelSpacing.map((value) => value.toFixed(3)).join(" × ")} mm`} />
                    <Field label="Orientation" value={<span className="font-mono">{metadata.orientation}</span>} />
                    <Field label="Datatype" value={`${metadata.datatype} (${metadata.datatypeCode})`} />
                  </dl>
                  <div className="border-t py-3">
                    <p className="mb-2 text-[11px] font-medium text-muted-foreground">Affine 4 × 4</p>
                    <div className="overflow-x-auto rounded-md border bg-muted/35 p-2">
                      {metadata.affine.length ? (
                        <table className="w-full border-collapse font-mono text-[10px] tabular-nums">
                          <tbody>
                            {metadata.affine.slice(0, 4).map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                {row.slice(0, 4).map((value, columnIndex) => (
                                  <td key={columnIndex} className="px-1.5 py-1 text-right">{Number(value).toFixed(3)}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <p className="text-xs text-muted-foreground">Матрица отсутствует в заголовке.</p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </Panel>
          </aside>
        </div>

        <footer className="flex flex-col gap-2 px-1 pb-3 pt-5 text-[10px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span className="flex items-center gap-1.5"><ShieldCheck className="h-3 w-3" /> MRI и CSV обрабатываются локально, без передачи на сервер.</span>
          <span className="flex items-center gap-1">Powered by Niivue <ChevronRight className="h-3 w-3" /> WebGL2</span>
        </footer>
      </main>
    </div>
  );
}
