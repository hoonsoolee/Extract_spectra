import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [workbookPath, previewDir] = process.argv.slice(2);
if (!workbookPath || !previewDir) {
  throw new Error("Usage: node verify_team_workbook.mjs workbook.xlsx preview_dir");
}

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const fieldCheck = await workbook.inspect({
  kind: "table",
  range: "Field Summary!A1:V8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 22,
});
const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
const drawingCheck = await workbook.inspect({
  kind: "drawing",
  sheetId: "Dashboard",
  maxChars: 3000,
});
console.log(fieldCheck.ndjson);
console.log(errorCheck.ndjson);
console.log(drawingCheck.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of [
  "Dashboard", "README", "Field Summary", "Cluster Summary",
  "Reflectance Spectra", "Warnings",
]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}
