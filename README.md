# email-synthesizer
Email Synthesizer is a tool for transforming real business emails into synthetic versions for AI training, testing, and demonstration purposes. It preserves the semantic intent and structure of original communications while replacing all personally identifiable information (PII) and company-specific references.

Email Synthesizer - Documentation

Prerequisites
Python 3.9 or higher
OpenAI API key (for LLM-based transformations)

Installation
Clone the repository: 
git clone https://github.com/yourusername/email-synthesizer.git

Navigate to the project directory:
cd email-synthesizer

Install required packages:
pip install -r requirements.txt

Running the Application

Start the Streamlit app:
streamlit run app.py

Using the Application-

Configure Settings:

Enter your OpenAI API key in the sidebar
Adjust transformation settings as needed (company name, location, industry, etc.)

Input Data:

Choose between "Direct Email Input" or "Upload File" tabs
For direct input: Paste email text directly into the text area
For file upload: Upload a CSV or Excel file containing email data

Generate Synthetic Emails:

Click "Generate Synthetic Email" (for direct input) or select emails and click "Generate Synthetic Emails" (for file upload)
Wait for processing to complete

Review Results:

Compare original and synthetic emails side-by-side
Check highlighted changes (color-coded by type)
Review validation results for any issues

Export Data (optional):
Click "Export as CSV" to download the results
