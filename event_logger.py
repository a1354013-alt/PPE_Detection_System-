import os
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass
class ViolationEvent:
    timestamp: str
    source: str
    track_id: str
    person_count: int
    missing_items: str
    screenshot_path: str
    confidence: float
    bbox: str
    event_type: str = "ppe_violation"
    category: str = "ppe"
    severity: str = "medium"
    details: str = ""
    region_name: str = ""
    threshold: int = 0
    frame_index: int = 0
    is_demo: bool = False


class EventLogger:
    def __init__(self):
        self.events: List[ViolationEvent] = []

    @staticmethod
    def _normalize_processing_summary(processing_summary):
        if not processing_summary:
            return {}
        if isinstance(processing_summary, dict):
            return processing_summary
        return {"summary_text": str(processing_summary)}

    @staticmethod
    def _ensure_output_dir(filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    @staticmethod
    def _format_event_time(timestamp: str):
        if " " in timestamp:
            return timestamp.split(" ", 1)[1]
        return timestamp

    def add_event(self, event: ViolationEvent):
        self.events.append(event)

    def clear_events(self):
        self.events.clear()

    def to_dataframe(self):
        if not self.events:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "source",
                    "track_id",
                    "person_count",
                    "missing_items",
                    "screenshot_path",
                    "confidence",
                    "bbox",
                    "event_type",
                    "category",
                    "severity",
                    "details",
                    "region_name",
                    "threshold",
                    "frame_index",
                    "is_demo",
                ]
            )
        return pd.DataFrame([asdict(event) for event in self.events])

    @staticmethod
    def _event_type_counts(events):
        counts = {}
        for event in events:
            event_type = getattr(event, "event_type", "ppe_violation") or "ppe_violation"
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def export_csv(self, filepath: str):
        self._ensure_output_dir(filepath)
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        return filepath

    def export_excel(self, filepath: str, stats: dict, processing_summary=None):
        self._ensure_output_dir(filepath)
        df = self.to_dataframe()
        processing_summary = self._normalize_processing_summary(processing_summary)
        wb = Workbook()

        ws1 = wb.active
        ws1.title = "Violation Details"
        for row in dataframe_to_rows(df, index=False, header=True):
            ws1.append(row)

        for column in ws1.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except (TypeError, ValueError) as exc:
                    logging.warning("Unable to size Excel column %s: %s", column_letter, exc)
            ws1.column_dimensions[column_letter].width = max_length + 2

        ws2 = wb.create_sheet(title="Summary")
        ws2.append(["Category", "Count"])
        ws2.append(["Total Safety Events", len(self.events)])
        for event_type, count in self._event_type_counts(self.events).items():
            ws2.append([f"{event_type} Events", count])
        for item, count in stats.get("missing_counts", {}).items():
            ws2.append([f"{item.capitalize()} Violations", count])
        for region_name, count in stats.get("crowd_region_counts", {}).items():
            ws2.append([f"Crowd Region: {region_name}", count])

        if processing_summary:
            ws2.append([])
            ws2.append(["Processing Summary", "Value"])
            for key, value in processing_summary.items():
                label = str(key).replace("_", " ").title()
                ws2.append([label, value])

        wb.save(filepath)
        return filepath

    def export_pdf(self, filepath: str, stats: dict, processing_summary=None):
        self._ensure_output_dir(filepath)
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []
        processing_summary = self._normalize_processing_summary(processing_summary)

        elements.append(Paragraph("Safety Event Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        elements.append(Paragraph(f"Total Events: {len(self.events)}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        summary_data = [["Category", "Count"]]
        for event_type, count in self._event_type_counts(self.events).items():
            summary_data.append([f"{event_type} Events", str(count)])
        for item, count in stats.get("missing_counts", {}).items():
            summary_data.append([f"{item.capitalize()} Violations", str(count)])
        for region_name, count in stats.get("crowd_region_counts", {}).items():
            summary_data.append([f"Crowd Region: {region_name}", str(count)])

        summary_table = Table(summary_data, colWidths=[200, 100])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(Paragraph("Summary Statistics", styles["Heading2"]))
        elements.append(summary_table)
        elements.append(Spacer(1, 24))

        if processing_summary:
            processing_data = [["Field", "Value"]]
            for key, value in processing_summary.items():
                processing_data.append([str(key).replace("_", " ").title(), str(value)])

            processing_table = Table(processing_data, colWidths=[180, 280])
            processing_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            elements.append(Paragraph("Processing Summary", styles["Heading2"]))
            elements.append(processing_table)
            elements.append(Spacer(1, 24))

        elements.append(Paragraph("Event Details (Top 50)", styles["Heading2"]))
        detail_rows = [["Time", "Type", "Severity", "Details"]]
        for event in self.events[:50]:
            details = event.details or event.missing_items or "-"
            detail_rows.append(
                [
                    self._format_event_time(event.timestamp),
                    event.event_type,
                    event.severity,
                    details,
                ]
            )

        detail_table = Table(detail_rows, colWidths=[70, 100, 70, 220])
        detail_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(detail_table)

        if self.events:
            elements.append(Spacer(1, 24))
            elements.append(Paragraph("Recent Violation Screenshots", styles["Heading2"]))
            for event in self.events[-5:]:
                if event.screenshot_path and os.path.exists(event.screenshot_path):
                    try:
                        image = Image(event.screenshot_path, width=400, height=250)
                        elements.append(Paragraph(f"Time: {event.timestamp} | ID: {event.track_id}", styles["Normal"]))
                        elements.append(image)
                        elements.append(Spacer(1, 12))
                    except (OSError, ValueError) as exc:
                        logging.warning("Unable to add screenshot to PDF report: %s", exc)

        doc.build(elements)
        return filepath
