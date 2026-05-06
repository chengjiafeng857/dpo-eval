#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Paragraph:
    text: str


@dataclass
class Table:
    rows: list[list[str]]


Block = Heading | Paragraph | Table


INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def parse_markdown(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    paragraph_lines: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines).strip()
            if text:
                blocks.append(Paragraph(text=text))
            paragraph_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            blocks.append(Heading(level=len(heading_match.group(1)), text=heading_match.group(2).strip()))
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows: list[list[str]] = []
            for raw in table_lines:
                cells = [cell.strip() for cell in raw.strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    continue
                rows.append(cells)
            if rows:
                blocks.append(Table(rows=rows))
            continue

        paragraph_lines.append(stripped)
        i += 1

    flush_paragraph()
    return blocks


def xml_text(text: str) -> str:
    return escape(text)


def runs_xml(text: str, code_style: bool = False, bold: bool = False) -> str:
    run_props = []
    if bold:
        run_props.append("<w:b/>")
    if code_style:
        run_props.append("<w:rFonts w:ascii=\"Courier New\" w:hAnsi=\"Courier New\"/>")
        run_props.append("<w:sz w:val=\"19\"/>")
    rpr = f"<w:rPr>{''.join(run_props)}</w:rPr>" if run_props else ""
    preserve = " xml:space=\"preserve\"" if text[:1] == " " or text[-1:] == " " else ""
    return f"<w:r>{rpr}<w:t{preserve}>{xml_text(text)}</w:t></w:r>"


def inline_runs_xml(text: str, bold: bool = False) -> str:
    parts: list[str] = []
    last = 0
    for match in INLINE_CODE_RE.finditer(text):
        if match.start() > last:
            parts.append(runs_xml(text[last:match.start()], bold=bold))
        parts.append(runs_xml(match.group(1), code_style=True, bold=bold))
        last = match.end()
    if last < len(text):
        parts.append(runs_xml(text[last:], bold=bold))
    if not parts:
        parts.append(runs_xml("", bold=bold))
    return "".join(parts)


def paragraph_xml(text: str, style: str | None = None, spacing_after: int = 120) -> str:
    props: list[str] = []
    if style:
        props.append(f"<w:pStyle w:val=\"{style}\"/>")
    props.append(f"<w:spacing w:after=\"{spacing_after}\"/>")
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>"
    return f"<w:p>{ppr}{inline_runs_xml(text)}</w:p>"


def heading_style(level: int) -> str:
    return {
        1: "Title",
        2: "Heading1",
        3: "Heading2",
        4: "Heading3",
    }.get(level, "Heading3")


def heading_spacing(level: int) -> int:
    return {
        1: 240,
        2: 160,
        3: 120,
        4: 120,
    }.get(level, 120)


def cell_xml(text: str, width: int, header: bool = False) -> str:
    shading = "<w:shd w:val=\"clear\" w:fill=\"EAEAEA\"/>" if header else ""
    content = (
        f"<w:p><w:pPr><w:spacing w:after=\"40\"/></w:pPr>{inline_runs_xml(text, bold=header)}</w:p>"
    )
    return (
        "<w:tc>"
        f"<w:tcPr><w:tcW w:w=\"{width}\" w:type=\"dxa\"/>{shading}<w:vAlign w:val=\"center\"/></w:tcPr>"
        f"{content}"
        "</w:tc>"
    )


def table_xml(rows: list[list[str]]) -> str:
    col_count = max(len(row) for row in rows)
    page_width = 9360
    widths = [page_width // col_count for _ in range(col_count)]

    grid = "".join(f"<w:gridCol w:w=\"{width}\"/>" for width in widths)
    row_xml: list[str] = []
    for row_idx, row in enumerate(rows):
        padded = row + [""] * (col_count - len(row))
        cells = "".join(
            cell_xml(text=cell, width=widths[col_idx], header=row_idx == 0)
            for col_idx, cell in enumerate(padded)
        )
        row_xml.append("<w:tr>" + cells + "</w:tr>")

    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblInd w:w=\"0\" w:type=\"dxa\"/>"
        "<w:tblLayout w:type=\"fixed\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"C9C9C9\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"C9C9C9\"/>"
        "</w:tblBorders>"
        "<w:tblCellMar>"
        "<w:top w:w=\"80\" w:type=\"dxa\"/>"
        "<w:left w:w=\"110\" w:type=\"dxa\"/>"
        "<w:bottom w:w=\"80\" w:type=\"dxa\"/>"
        "<w:right w:w=\"110\" w:type=\"dxa\"/>"
        "</w:tblCellMar>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{''.join(row_xml)}"
        "</w:tbl>"
        "<w:p><w:pPr><w:spacing w:after=\"120\"/></w:pPr></w:p>"
    )


def document_xml(blocks: list[Block]) -> str:
    body: list[str] = []
    for block in blocks:
        if isinstance(block, Heading):
            body.append(
                paragraph_xml(
                    text=block.text,
                    style=heading_style(block.level),
                    spacing_after=heading_spacing(block.level),
                )
            )
        elif isinstance(block, Paragraph):
            body.append(paragraph_xml(text=block.text))
        else:
            body.append(table_xml(block.rows))

    sect_pr = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1080\" w:right=\"900\" w:bottom=\"1080\" w:left=\"900\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        f"<w:document xmlns:w=\"{W_NS}\">"
        f"<w:body>{''.join(body)}{sect_pr}</w:body>"
        "</w:document>"
    )


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:lang w:val="en-US"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="34"/><w:color w:val="1F1F1F"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="200" w:after="140"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="244061"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="140" w:after="100"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="355C7D"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="4F81BD"/></w:rPr>
  </w:style>
</w:styles>
"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""


def core_xml(title: str) -> str:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xml_text(title)}</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>
"""


def write_docx(markdown_path: Path, docx_path: Path) -> None:
    blocks = parse_markdown(markdown_path.read_text(encoding="utf-8"))
    title = next((b.text for b in blocks if isinstance(b, Heading)), markdown_path.stem)
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", rels_xml())
        zf.writestr("word/document.xml", document_xml(blocks))
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml())
        zf.writestr("docProps/core.xml", core_xml(title))
        zf.writestr("docProps/app.xml", app_xml())


def main() -> int:
    root = Path(__file__).resolve().parent / "result"
    for stem in ("qwen_hh_sweeps_report", "llama_hh_sweeps_report"):
        md_path = root / f"{stem}.md"
        docx_path = root / f"{stem}.docx"
        write_docx(md_path, docx_path)
        print(docx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
