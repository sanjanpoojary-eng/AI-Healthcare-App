import streamlit as st
import pandas as pd
from io import BytesIO

import pdf_worlds   # Your PDF extraction module
import chatbot      # Your chatbot module

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch   # ← for margin calculations

# --- Streamlit session state setup ---
if "findings_df" not in st.session_state:
    st.session_state.findings_df = None
if "findings_md" not in st.session_state:
    st.session_state.findings_md = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Healthcare Report Analyzer")

# --- Sidebar: PDF upload ---
st.sidebar.header("Upload Healthcare Report")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    pdf_text = pdf_worlds.extract_text(uploaded_file.read())
    prompt_findings = f"""
Extract all important findings from this healthcare report and present them in a structured table format with columns such as 'Finding', 'Details', and 'Recommendations'. Output in markdown table format.

{pdf_text}
"""
    with st.spinner("Analyzing report..."):
        findings_md = chatbot.generate_response(prompt_findings)
        st.session_state.findings_md = findings_md
        st.subheader("Extracted Key Findings")

        # Parse markdown table to DataFrame
        try:
            start = findings_md.find('|')
            end   = findings_md.rfind('|') + 1
            table_md = findings_md[start:end]
            lines = table_md.strip().split('\n')
            headers = [h.strip() for h in lines[0].split('|')[1:-1]]
            rows = []
            for line in lines[2:]:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if cols:
                    rows.append(cols)
            df = pd.DataFrame(rows, columns=headers)
            st.session_state.findings_df = df
            st.table(df)
        except Exception as e:
            st.error(f"Error parsing table: {e}")
            st.markdown(findings_md)
else:
    st.info("Please upload a healthcare report in PDF format via the sidebar.")

st.markdown("---")
st.header("Chat with the Healthcare Assistant")

# --- Chat form ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", "")
    send = st.form_submit_button("Send")
    if send and user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        if uploaded_file and st.session_state.findings_md:
            context = f"""
You are a healthcare assistant. Based on the following key findings from a healthcare report, please answer the user's question.

Key findings:
{st.session_state.findings_md}

User's question: {user_input}
"""
        else:
            context = user_input

        with st.spinner("Generating response..."):
            resp = chatbot.generate_response(context)
            st.session_state.chat_history.append({"role": "assistant", "message": resp})

            if uploaded_file and st.session_state.findings_md:
                valid = chatbot.validate_response(
                    st.session_state.findings_md,
                    user_input,
                    resp
                )
                st.session_state.chat_history.append({"role": "validator", "message": valid})

# --- Display chat history ---
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['message']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['message']}")
    else:
        st.markdown(f"**Validation:** {msg['message']}")

# --- PDF summary generator ---
def generate_pdf_summary(findings_df, chat_history):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Healthcare Report Summary", styles['Title']))
    elements.append(Spacer(1, 12))

    # Findings table
    if findings_df is not None:
        elements.append(Paragraph("Extracted Key Findings", styles['Heading2']))
        elements.append(Spacer(1, 6))

        h_style = styles['Heading4']
        b_style = styles['BodyText']

        # Build table data
        table_data = [
            [Paragraph(col, h_style) for col in findings_df.columns]
        ]
        for row in findings_df.itertuples(index=False):
            table_data.append([
                Paragraph(str(cell), b_style) for cell in row
            ])

        # Calculate column widths
        col_widths = []
        for ci in range(len(table_data[0])):
            max_chars = max(len(table_data[ri][ci].text) for ri in range(len(table_data)))
            col_widths.append(max_chars * 6)
        total_w = sum(col_widths)
        avail_w = letter[0] - 2 * inch
        if total_w > avail_w:
            scale = avail_w / total_w
            col_widths = [w * scale for w in col_widths]

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        h = tbl.wrap(doc.width, doc.height)[1]
        if h > doc.height:
            elements.append(tbl)
            elements.append(PageBreak())
        else:
            elements.append(tbl)
        elements.append(Spacer(1, 12))

    # Chat history
    elements.append(Paragraph("Chat History", styles['Heading2']))
    elements.append(Spacer(1, 6))
    for m in chat_history:
        prefix = {"user":"You","assistant":"Assistant","validator":"Validation"}[m["role"]]
        elements.append(Paragraph(f"<b>{prefix}:</b> {m['message']}", styles['Normal']))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# --- Download button ---
if st.session_state.findings_df is not None and st.session_state.chat_history:
    pdf_bytes = generate_pdf_summary(
        st.session_state.findings_df,
        st.session_state.chat_history
    )
    st.download_button(
        label="Download Summary PDF",
        data=pdf_bytes,
        file_name="summary.pdf",
        mime="application/pdf"
    )
