export type PhenotypicRecord = Record<string, string>;

export function normalizeSubjectId(value?: string | number | null) {
  const raw = String(value ?? "").trim();
  if (!raw) return "";
  const numeric = raw.replace(/\.0+$/, "").replace(/^0+(?=\d)/, "");
  return numeric || "0";
}

export function inferSubjectId(fileName: string) {
  const candidates = fileName.match(/\d{5,}/g);
  return candidates?.at(-1) ?? "";
}

export function normalizeRecord(record: Record<string, unknown>): PhenotypicRecord {
  return Object.fromEntries(
    Object.entries(record).map(([key, value]) => [
      key.replace(/^\uFEFF/, "").trim().toUpperCase(),
      String(value ?? "").trim(),
    ]),
  );
}

export function getDiagnosis(record?: PhenotypicRecord) {
  const value = record?.DX_GROUP?.trim().toLowerCase();
  if (!value) return null;
  if (value === "1" || value.includes("aut") || value.includes("asd")) return "ASD";
  if (value === "2" || value.includes("control") || value === "td") return "Control";
  return record?.DX_GROUP ?? null;
}

export function getSex(record?: PhenotypicRecord) {
  const value = record?.SEX?.trim().toLowerCase();
  if (!value) return null;
  if (value === "1" || value === "m" || value === "male") return "Мужской";
  if (value === "2" || value === "f" || value === "female") return "Женский";
  return record?.SEX ?? null;
}
