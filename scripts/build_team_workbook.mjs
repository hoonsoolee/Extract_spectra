import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath, previewDir] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("Usage: node build_team_workbook.mjs input.json output.xlsx [preview_dir]");
}

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
const readme = workbook.worksheets.add("README");
const field = workbook.worksheets.add("Field Summary");
const clusters = workbook.worksheets.add("Cluster Summary");
const spectra = workbook.worksheets.add("Reflectance Spectra");
const warnings = workbook.worksheets.add("Warnings");
for (const sheet of [dashboard, readme, field, clusters, spectra, warnings]) {
  sheet.showGridLines = false;
}

const green = "#176848";
const darkGreen = "#173F35";
const paleGreen = "#EAF5F0";
const lightLine = "#D9E5DF";
const amber = "#FFF2CC";
const red = "#FCE8E6";
const gray = "#EEF1F0";

function safeDate(value) {
  if (!value || String(value).startsWith("Unspecified")) return null;
  const parsed = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function addTable(sheet, name, headers, rows) {
  const values = [headers, ...rows];
  const range = sheet.getRangeByIndexes(0, 0, values.length, headers.length);
  range.values = values;
  const header = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  header.format = {
    fill: darkGreen,
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "medium", color: darkGreen },
  };
  header.format.rowHeight = 30;
  if (rows.length) {
    const body = sheet.getRangeByIndexes(1, 0, rows.length, headers.length);
    body.format.borders = {
      insideHorizontal: { style: "thin", color: lightLine },
      bottom: { style: "thin", color: lightLine },
    };
  }
  sheet.tables.add(range.address, true, name);
  sheet.freezePanes.freezeRows(1);
  return range;
}

const meta = payload.meta || {};
const summaryRows = payload.summaries || [];
const fieldHeaders = [
  "Measurement Date", "Team", "Plot ID", "Filename", "Treatment", "Genotype",
  "Replicate", "Value Units", "Calibration QC", "Included", "NDVI Mean",
  "NDVI Median", "NDVI Q25", "NDVI Q75", "NDVI IQR", "Vegetation Fraction",
  "Classes", "Silhouette", "Davies-Bouldin", "Elapsed (s)", "Detail Report", "Source File",
];
const fieldRows = summaryRows.map((row) => [
  safeDate(row.measurement_date), row.team || "", row.plot_id || "", row.filename || "",
  row.treatment || "", row.genotype || "", row.replicate || "", row.value_units || "",
  row.calibration_qc_status || "", row.included_in_team_statistics ? "Yes" : "No",
  row.ndvi_mean ?? null, row.ndvi_median ?? null, row.ndvi_q25 ?? null, row.ndvi_q75 ?? null,
  null, row.vegetation_fraction ?? null, row.n_classes ?? null, row.silhouette ?? null,
  row.davies_bouldin ?? null, row.elapsed_seconds ?? null, row.detail_report || "", row.source_file || "",
]);
addTable(field, "FieldSummaryTable", fieldHeaders, fieldRows);
if (fieldRows.length) {
  field.getRange(`O2`).formulas = [[`=IF(OR(M2="",N2=""),"",N2-M2)`]];
  field.getRange(`O2:O${fieldRows.length + 1}`).fillDown();
  field.getRange(`A2:A${fieldRows.length + 1}`).format.numberFormat = "yyyy-mm-dd";
  field.getRange(`K2:O${fieldRows.length + 1}`).format.numberFormat = "0.0000";
  field.getRange(`P2:P${fieldRows.length + 1}`).format.numberFormat = "0.0%";
  field.getRange(`Q2:Q${fieldRows.length + 1}`).format.numberFormat = "#,##0";
  field.getRange(`R2:T${fieldRows.length + 1}`).format.numberFormat = "0.000";
  const qcRange = field.getRange(`I2:I${fieldRows.length + 1}`);
  qcRange.conditionalFormats.add("containsText", { text: "PASS", format: { fill: "#D9EAD3", font: { color: "#1D5E32", bold: true } } });
  qcRange.conditionalFormats.add("containsText", { text: "REVIEW", format: { fill: amber, font: { color: "#8A5A00", bold: true } } });
  qcRange.conditionalFormats.add("containsText", { text: "FAIL", format: { fill: red, font: { color: "#A12622", bold: true } } });
  qcRange.conditionalFormats.add("containsText", { text: "UNASSESSED", format: { fill: gray, font: { color: "#555555" } } });
}
field.getRange("A:V").format.autofitColumns();
for (const col of ["D", "E", "F", "U", "V"]) field.getRange(`${col}:${col}`).format.columnWidth = col === "V" ? 42 : 24;

const clusterHeaders = [
  "Measurement Date", "Team", "Plot ID", "Filename", "Class ID", "Class Name",
  "Pixel Count", "Fraction",
];
const clusterRows = (payload.cluster_rows || []).map((row) => [
  safeDate(row.measurement_date), row.team || "", row.plot_id || "", row.filename || "",
  row.class_id ?? null, row.class_name || "", row.pixel_count ?? null, row.fraction ?? null,
]);
addTable(clusters, "ClusterSummaryTable", clusterHeaders, clusterRows);
if (clusterRows.length) {
  clusters.getRange(`A2:A${clusterRows.length + 1}`).format.numberFormat = "yyyy-mm-dd";
  clusters.getRange(`G2:G${clusterRows.length + 1}`).format.numberFormat = "#,##0";
  clusters.getRange(`H2:H${clusterRows.length + 1}`).format.numberFormat = "0.0%";
}
clusters.getRange("A:H").format.autofitColumns();
clusters.getRange("D:D").format.columnWidth = 26;

const spectraHeaders = [
  "Measurement Date", "Team", "Plot ID", "Filename", "Class ID", "Class Name",
  "Pixel Count", "Wavelength (nm)", "Mean", "Median", "Q25", "Q75",
];
const spectraRows = (payload.spectra_rows || []).map((row) => [
  safeDate(row.measurement_date), row.team || "", row.plot_id || "", row.filename || "",
  row.class_id ?? null, row.class_name || "", row.pixel_count ?? null,
  row.wavelength_nm ?? null, row.mean ?? null, row.median ?? null, row.q25 ?? null, row.q75 ?? null,
]);
addTable(spectra, "ReflectanceSpectraTable", spectraHeaders, spectraRows);
if (spectraRows.length) {
  spectra.getRange(`A2:A${spectraRows.length + 1}`).format.numberFormat = "yyyy-mm-dd";
  spectra.getRange(`G2:G${spectraRows.length + 1}`).format.numberFormat = "#,##0";
  spectra.getRange(`H2:H${spectraRows.length + 1}`).format.numberFormat = "0.0";
  spectra.getRange(`I2:L${spectraRows.length + 1}`).format.numberFormat = "0.000000";
}
spectra.getRange("A:L").format.autofitColumns();
spectra.getRange("D:D").format.columnWidth = 26;

const warningHeaders = ["Measurement Date", "Team", "Plot ID", "Severity", "Code", "Message"];
const warningRows = (payload.warnings || []).map((row) => [
  safeDate(row.measurement_date), row.team || "", row.plot_id || "", row.severity || "",
  row.code || "", row.message || "",
]);
addTable(warnings, "WarningsTable", warningHeaders, warningRows);
if (warningRows.length) warnings.getRange(`A2:A${warningRows.length + 1}`).format.numberFormat = "yyyy-mm-dd";
warnings.getRange("A:F").format.autofitColumns();
warnings.getRange("F:F").format.columnWidth = 62;
warnings.getRange("F:F").format.wrapText = true;

readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [["Hyperspectral Team/Day Results"]];
readme.getRange("A1:F1").format = { fill: darkGreen, font: { bold: true, color: "#FFFFFF", size: 18 }, verticalAlignment: "center" };
readme.getRange("A1:F1").format.rowHeight = 36;
const readmeValues = [
  ["Field", "Value"],
  ["Measurement date", meta.measurement_date || ""],
  ["Team", meta.team || ""],
  ["Generated at", meta.generated_at || ""],
  ["Inclusion rule", meta.inclusion_rule || ""],
  ["Field Summary", "One row per plot; NDVI IQR is formula-derived from Q75-Q25."],
  ["Cluster Summary", "Pixel count and fraction for each plot-level cluster."],
  ["Reflectance Spectra", "Only science-ready plots with calibrated reflectance and calibration QC PASS."],
  ["Warnings", "Plots excluded from team statistics and the reason."],
  ["Important", "Do not pool image pixels across plots. Each plot is the statistical unit."],
];
readme.getRange(`A3:B${readmeValues.length + 2}`).values = readmeValues;
readme.getRange("A3:B3").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
readme.getRange(`A4:A${readmeValues.length + 2}`).format.font = { bold: true, color: darkGreen };
readme.getRange(`A3:B${readmeValues.length + 2}`).format.borders = { preset: "inside", style: "thin", color: lightLine };
readme.getRange("A:B").format.autofitColumns();
readme.getRange("B:B").format.columnWidth = 72;
readme.getRange("B:B").format.wrapText = true;

dashboard.getRange("A1:H2").merge();
dashboard.getRange("A1").values = [[`${meta.team || "Team"} · ${meta.measurement_date || "Date"} Daily Field Summary`]];
dashboard.getRange("A1:H2").format = { fill: darkGreen, font: { bold: true, color: "#FFFFFF", size: 18 }, verticalAlignment: "center" };
dashboard.getRange("A4:B4").values = [["KPI", "Value"]];
dashboard.getRange("A4:B4").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
dashboard.getRange("A5:A8").values = [["Total plots"], ["Calibration QC PASS"], ["Included in NDVI summary"], ["Mean of plot NDVI medians"]];
const endRow = Math.max(2, fieldRows.length + 1);
dashboard.getRange("B5:B8").formulas = [
  [`=COUNTA('Field Summary'!$C$2:$C$${endRow})`],
  [`=COUNTIF('Field Summary'!$I$2:$I$${endRow},"PASS")`],
  [`=COUNTIF('Field Summary'!$J$2:$J$${endRow},"Yes")`],
  [`=IFERROR(AVERAGEIF('Field Summary'!$J$2:$J$${endRow},"Yes",'Field Summary'!$L$2:$L$${endRow}),"")`],
];
dashboard.getRange("B5:B7").format.numberFormat = "#,##0";
dashboard.getRange("B8").format.numberFormat = "0.000";
dashboard.getRange("A4:B8").format.borders = { preset: "outside", style: "thin", color: lightLine };
dashboard.getRange("A5:A8").format = { fill: paleGreen, font: { bold: true, color: darkGreen } };

dashboard.getRange("A10:B10").values = [["Plot ID", "NDVI Median (QC PASS)"]];
dashboard.getRange("A10:B10").format = { fill: green, font: { bold: true, color: "#FFFFFF" } };
if (fieldRows.length) {
  const helper = [];
  for (let index = 0; index < fieldRows.length; index += 1) {
    const sourceRow = index + 2;
    helper.push([
      `='Field Summary'!C${sourceRow}`,
      `=IF('Field Summary'!J${sourceRow}="Yes",'Field Summary'!L${sourceRow},"")`,
    ]);
  }
  dashboard.getRange(`A11:B${fieldRows.length + 10}`).formulas = helper;
  dashboard.getRange(`B11:B${fieldRows.length + 10}`).format.numberFormat = "0.000";
  const chart = dashboard.charts.add("bar", dashboard.getRange(`A10:B${fieldRows.length + 10}`));
  chart.title = "Plot-level NDVI median (calibration QC PASS only)";
  chart.hasLegend = false;
  chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart.yAxis = { numberFormatCode: "0.00", min: -1, max: 1 };
  chart.setPosition("D4", "L22");
}
dashboard.getRange("A:B").format.columnWidth = 28;
dashboard.freezePanes.freezeRows(2);

const keyCheck = await workbook.inspect({
  kind: "table",
  range: `Field Summary!A1:V${Math.min(endRow, 12)}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 22,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "team workbook formula error scan",
});
console.log(keyCheck.ndjson);
console.log(errors.ndjson);

if (previewDir) {
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of ["Dashboard", "README", "Field Summary", "Cluster Summary", "Reflectance Spectra", "Warnings"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const fileName = `${sheetName.replaceAll(" ", "_")}.png`;
    await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
