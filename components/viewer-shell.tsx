"use client";

import dynamic from "next/dynamic";

const MRIViewer = dynamic(() => import("@/components/mri-viewer"), {
  ssr: false,
  loading: () => (
    <main className="grid min-h-screen place-items-center bg-background px-6">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="relative h-12 w-12 overflow-hidden rounded-xl border border-primary/20 bg-primary/10">
          <div className="absolute inset-0 animate-shimmer bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
        </div>
        <div>
          <p className="font-semibold">Инициализация MRI Viewer</p>
          <p className="mt-1 text-sm text-muted-foreground">Подготавливаем WebGL и Niivue…</p>
        </div>
      </div>
    </main>
  ),
});

export function ViewerShell() {
  return <MRIViewer />;
}
