# Email Synthesizer

## Overview
Email Synthesizer is a powerful tool designed to transform real business emails into synthetic versions for AI training, testing, and demonstrations. It ensures the preservation of semantic intent and structure while removing all personally identifiable information (PII) and company-specific references.

## Features
- **LLM-Based Transformation:** Uses OpenAI's language models to generate synthetic email versions.
- **Privacy Protection:** Automatically removes and replaces sensitive information.
- **Flexible Input Options:** Supports direct email input and bulk processing via file uploads.
- **Side-by-Side Comparison:** View original and synthetic emails with highlighted changes.
- **Customizable Settings:** Adjust transformation parameters, such as company names, locations, and industries.
- **Exportable Results:** Save transformed emails as CSV files for further use.

## Prerequisites
- Python 3.9 or higher
- OpenAI API key (for LLM-based transformations)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/email-synthesizer.git
   ```
2. Navigate to the project directory:
   ```sh
   cd email-synthesizer
   ```
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Application
Start the Streamlit app with the following command:
```sh
streamlit run app.py
```

### Using the Application
#### 1. Configure Settings
- Enter your OpenAI API key in the sidebar.
- Adjust transformation settings as needed (e.g., company name, location, industry, etc.).

#### 2. Input Data
Choose one of the following input methods:
- **Direct Email Input:** Paste the email text directly into the provided text area.
- **File Upload:** Upload a CSV or Excel file containing multiple emails for batch processing.

#### 3. Generate Synthetic Emails
- Click **"Generate Synthetic Email"** (for direct input) or select emails and click **"Generate Synthetic Emails"** (for file uploads).
- Wait for processing to complete.

#### 4. Review Results
- View original and synthetic emails side-by-side.
- Changes are highlighted using a color-coded system.
- Review validation results to ensure accuracy.

#### 5. Export Data (Optional)
- Click **"Export as CSV"** to download the results.

## Contributing
Contributions are welcome! If you'd like to improve Email Synthesizer, feel free to fork the repository and submit a pull request.
