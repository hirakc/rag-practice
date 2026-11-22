"""Debug helper: show how Docling structures one PDF (headings, labels, a formula, a table).

    uv run python scripts/probe_docling.py data/papers/2005.11401.pdf
"""

import os
import sys
import time
from collections import Counter

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

opts = PdfPipelineOptions(
    do_formula_enrichment=os.environ.get("FORMULA", "1") == "1",
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.AUTO),
)
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

t = time.time()
res = conv.convert(sys.argv[1])
print("seconds:", round(time.time() - t, 1), res.status)
doc = res.document

labels = Counter()
for item, _ in doc.iterate_items():
    labels[item.label.value] += 1
    if item.label.value in ("section_header", "title"):
        page = item.prov[0].page_no if item.prov else None
        print("H", getattr(item, "level", None), page, item.text[:70])
print(labels)

for item, _ in doc.iterate_items():
    if item.label.value == "formula":
        print("FORMULA:", item.text[:120])
        break
for item, _ in doc.iterate_items():
    if item.label.value == "table":
        print("TABLE caption:", item.caption_text(doc)[:100])
        print(item.export_to_markdown(doc)[:300])
        break
