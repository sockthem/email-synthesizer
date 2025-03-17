import streamlit as st
import pandas as pd
import re
import os
from io import StringIO
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import difflib
import html

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Email Synthesizer", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = []

# === UTILITY CLASSES ===
class EntityReplacer:
    def __init__(self, transformation_settings=None):
        """
        Initialize the entity replacer with custom transformation settings
        
        Args:
            transformation_settings (dict): Dictionary with transformation settings
        """
        # Default settings
        self.settings = {
            "original_company": "Enron",
            "new_company": "Agriculture India",
            "email_domain": "agriindia.com",
            "currency_conversion_rate": 80,
            "location": "India",
            "year": 2024
        }
        
        # Update with custom settings if provided
        if transformation_settings:
            self.settings.update(transformation_settings)
        
        # Regular expression patterns for replacements
        self.patterns = {
            fr'\b{self.settings["original_company"]}\b': self.settings["new_company"],
            r'@\w+\.enron\.com': f'@{self.settings["email_domain"]}',
        }
        
        # More complex patterns that need custom handling
        self.complex_patterns = [
            self.convert_currency,
            self.convert_dates,
            self.convert_phone_numbers,
        ]
    
    def replace(self, text):
        if not text or not isinstance(text, str):
            return text
            
        # Apply simple pattern replacements
        for pattern, repl in self.patterns.items():
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
        # Apply complex replacements
        for handler in self.complex_patterns:
            text = handler(text)
            
        return text
    
    def convert_currency(self, text):
        """
        Converts dollar amounts to the target currency with appropriate formatting
        based on the target location.
        """
        # Get conversion rate from settings
        conversion_rate = self.settings.get("currency_conversion_rate", 80)
        location = self.settings.get("location", "India")
        
        # Pattern for currency with multipliers (million, billion, etc.)
        pattern_with_multiplier = r'\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|m|b|k|thousand)'
        
        def replace_with_multiplier(match):
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            multiplier = match.group(2).lower()
            
            # Apply multiplier
            if multiplier in ('billion', 'b'):
                amount *= 1000000000
            elif multiplier in ('million', 'm'):
                amount *= 1000000
            elif multiplier in ('thousand', 'k'):
                amount *= 1000
            
            # Convert to target currency
            converted_amount = amount * conversion_rate
            
            # Format appropriately based on location
            if location == "India":
                # Use Indian number system (lakhs and crores)
                if converted_amount >= 10000000:  # More than 1 crore
                    crores = converted_amount / 10000000
                    return f"₹{crores:.0f} crore" if crores.is_integer() else f"₹{crores:.2f} crore"
                elif converted_amount >= 100000:  # More than 1 lakh
                    lakhs = converted_amount / 100000
                    return f"₹{lakhs:.0f} lakh" if lakhs.is_integer() else f"₹{lakhs:.2f} lakh"
                else:
                    return f"₹{int(converted_amount):,}"
            elif location == "UK":
                # Use GBP for UK
                return f"£{converted_amount:,.2f}" if converted_amount % 1 else f"£{int(converted_amount):,}"
            elif location in ["Australia", "Canada"]:
                # Use respective dollar symbols
                currency_symbol = "A$" if location == "Australia" else "C$"
                return f"{currency_symbol}{converted_amount:,.2f}" if converted_amount % 1 else f"{currency_symbol}{int(converted_amount):,}"
            elif location == "Singapore":
                return f"S${converted_amount:,.2f}" if converted_amount % 1 else f"S${int(converted_amount):,}"
            elif location == "South Africa":
                return f"R{converted_amount:,.2f}" if converted_amount % 1 else f"R{int(converted_amount):,}"
            else:
                # Default format (international)
                return f"${converted_amount:,.2f}" if converted_amount % 1 else f"${int(converted_amount):,}"
        
        # Replace currency with multipliers
        text = re.sub(pattern_with_multiplier, replace_with_multiplier, text, flags=re.IGNORECASE)
        
        # Pattern for simple currency amounts ($XXX.XX)
        pattern_simple = r'\$\s*([\d,]+(?:\.\d+)?)'
        
        def replace_simple(match):
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            converted_amount = amount * conversion_rate
            
            # Format appropriately based on location (same as above)
            if location == "India":
                if converted_amount >= 10000000:  # More than 1 crore
                    crores = converted_amount / 10000000
                    return f"₹{crores:.0f} crore" if crores.is_integer() else f"₹{crores:.2f} crore"
                elif converted_amount >= 100000:  # More than 1 lakh
                    lakhs = converted_amount / 100000
                    return f"₹{lakhs:.0f} lakh" if lakhs.is_integer() else f"₹{lakhs:.2f} lakh"
                else:
                    return f"₹{int(converted_amount):,}"
            elif location == "UK":
                return f"£{converted_amount:,.2f}" if converted_amount % 1 else f"£{int(converted_amount):,}"
            elif location in ["Australia", "Canada"]:
                currency_symbol = "A$" if location == "Australia" else "C$"
                return f"{currency_symbol}{converted_amount:,.2f}" if converted_amount % 1 else f"{currency_symbol}{int(converted_amount):,}"
            elif location == "Singapore":
                return f"S${converted_amount:,.2f}" if converted_amount % 1 else f"S${int(converted_amount):,}"
            elif location == "South Africa":
                return f"R{converted_amount:,.2f}" if converted_amount % 1 else f"R{int(converted_amount):,}"
            else:
                return f"${converted_amount:,.2f}" if converted_amount % 1 else f"${int(converted_amount):,}"
        
        # Replace simple currency amounts
        text = re.sub(pattern_simple, replace_simple, text)
        
        return text
    
    def convert_dates(self, text):
        """
        Updates dates to the target year while preserving month and day.
        Handles multiple date formats.
        """
        target_year = self.settings.get("year", 2024)
        
        # Handle ISO format dates (YYYY-MM-DD)
        iso_pattern = r'\b(19|20)\d{2}[-/](\d{2})[-/](\d{2})\b'
        
        def replace_date(match):
            # Keep month and day, replace year with target year
            return f"{target_year}-{match.group(2)}-{match.group(3)}"
        
        text = re.sub(iso_pattern, replace_date, text)
        
        # Handle written dates (January 15, 2001)
        months = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        written_pattern = f'\\b({months})\\s+(\\d{{1,2}}),\\s+(19|20)\\d{{2}}\\b'
        
        def replace_written_date(match):
            return f"{match.group(1)} {match.group(2)}, {target_year}"
        
        text = re.sub(written_pattern, replace_written_date, text, flags=re.IGNORECASE)
        
        return text
    
    def convert_phone_numbers(self, text):
        """
        Converts US phone numbers to the appropriate format for the target location.
        """
        location = self.settings.get("location", "India")
        
        # Define country codes for different locations
        country_codes = {
            "India": "+91",
            "UK": "+44",
            "Australia": "+61",
            "Canada": "+1",
            "Singapore": "+65",
            "South Africa": "+27"
        }
        
        country_code = country_codes.get(location, "+91")
        
        # Pattern to match US phone numbers like (412) 490-9048
        us_pattern = r'\((\d{3})\)\s*(\d{3})[-\s](\d{4})'
        
        def replace_phone(match):
            # Convert to target format based on location
            if location == "India":
                return f"{country_code}-11-{match.group(2)}-{match.group(3)}"
            elif location == "UK":
                return f"{country_code} 20 {match.group(2)} {match.group(3)}"
            else:
                return f"{country_code} {match.group(1)} {match.group(2)} {match.group(3)}"
        
        text = re.sub(us_pattern, replace_phone, text)
        
        # Pattern for other formats like 412-490-9048
        alt_pattern = r'(\d{3})[-\s](\d{3})[-\s](\d{4})'
        text = re.sub(alt_pattern, replace_phone, text)
        
        return text

class LLMParaphraser:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
    def paraphrase(self, text, metadata=None):
        if not self.client:
            return "API key required"
        
        # Default metadata if none provided
        if not metadata:
            metadata = {
                "original_company": "Enron",
                "new_company": "Agriculture India",
                "industry": "agriculture",
                "year": 2024,
                "location": "India",
                "currency_conversion_rate": 80,  # USD to INR
                "email_domain": "agriindia.com"
            }
            
        try:
            # Customize the email domain extension based on location
            email_domain = metadata.get("email_domain", "agriindia.com")
            
            # Create location-specific guidance
            location_guidance = {
                "India": {
                    "currency": "rupees (₹)",
                    "number_format": "Use 'lakh' for hundreds of thousands and 'crore' for tens of millions",
                    "business_entities": "Private Limited (Pvt. Ltd.), Limited (Ltd.)"
                },
                "UK": {
                    "currency": "pounds (£)",
                    "number_format": "Standard international format with commas",
                    "business_entities": "Limited (Ltd.), Public Limited Company (PLC)"
                },
                "Australia": {
                    "currency": "Australian dollars (A$)",
                    "number_format": "Standard international format with commas",
                    "business_entities": "Proprietary Limited (Pty Ltd)"
                },
                "Canada": {
                    "currency": "Canadian dollars (C$)",
                    "number_format": "Standard international format with commas",
                    "business_entities": "Limited (Ltd.), Incorporated (Inc.)"
                },
                "Singapore": {
                    "currency": "Singapore dollars (S$)",
                    "number_format": "Standard international format with commas",
                    "business_entities": "Private Limited (Pte Ltd)"
                },
                "South Africa": {
                    "currency": "rand (R)",
                    "number_format": "Standard international format with commas",
                    "business_entities": "Proprietary Limited (Pty Ltd)"
                }
            }
            
            # Get location-specific guidance
            location = metadata.get("location", "India")
            guidance = location_guidance.get(location, location_guidance["India"])
            
            system_prompt = f"""You are an expert in creating synthetic business email data for AI training purposes.

Your task is to transform source emails from {metadata['original_company']} into realistic business emails that appear to be from {metadata['new_company']}, a company in the {metadata['industry']} sector in {metadata['location']}.

Follow these specific guidelines:
1. Maintain the original communication intent, tone, and complexity
2. Preserve the overall structure, length, and detail level of the original
3. Transform ALL specific identifiers:
   - Replace ALL person names with culturally appropriate {metadata['location']} names
   - Replace {metadata['original_company']} with {metadata['new_company']} 
   - Convert ALL email addresses to @{email_domain} format
   - Change ALL dollar amounts to {guidance['currency']} with appropriate conversion:
     * For currency values: multiply USD by {metadata['currency_conversion_rate']}
     * {guidance['number_format']}
   - Update ALL dates to {metadata['year']} while maintaining seasonal relevance
   - Transform location references to appropriate {metadata['location']} locations
   - Replace ALL phone numbers with {metadata['location']} format
   - Change ALL industry-specific terms to {metadata['industry']} sector equivalents

4. Create plausible but fictional:
   - Replace project names with {metadata['industry']} sector equivalents
   - Change business entities to {metadata['location']} {metadata['industry']} companies using {guidance['business_entities']} formats
   - Adapt technical terms to {metadata['industry']} context
   - Transform financial metrics to {metadata['industry']} sector standards

5. Ensure the synthetic email reads naturally and authentically as if written by a real {metadata['new_company']} employee

Output only the transformed email with no explanations."""

            #user_prompt = f"Transform this {
            
            user_prompt = f"Transform this {metadata['original_company']} email into a realistic {metadata['new_company']} email:\n\n{text}"
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class Validator:
    def validate(self, text, original_company="Enron"):
        """Validates the transformed text for any missed transformations"""
        if not text or not isinstance(text, str):
            return ["Empty or invalid text"]
            
        issues = []
        # Check for original company name
        if re.search(fr'\b{original_company}\b', text, re.IGNORECASE):
            issues.append(f"Unreplaced company name: '{original_company}'")
            
        # Check for Enron email domains
        if re.search(r'@\w+\.enron\.com', text, re.IGNORECASE):
            issues.append("Unreplaced email domain")
            
        # Check for dollar signs
        if re.search(r'\$', text):
            issues.append("Unreplaced dollar sign")
            
        # Check for US-style phone numbers
        if re.search(r'\(\d{3}\)\s*\d{3}-\d{4}', text) or re.search(r'\b\d{3}-\d{3}-\d{4}\b', text):
            issues.append("Unreplaced US phone number")
            
        # Check for old dates (before 2023)
        date_patterns = [
            r'\b(19|20[0-1]\d|202[0-2])[-/]\d{2}[-/]\d{2}\b',  # ISO format
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+(19|20[0-1]\d|202[0-2])\b'  # Written format
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Contains dates before 2023")
                break
                
        return issues

# === HIGHLIGHTING AND DISPLAY FUNCTIONS ===
def highlight_changes(original_text, synthetic_text, transformation_settings):
    """
    Adds HTML color highlighting to the synthetic text to indicate changes
    
    Args:
        original_text (str): The original email text
        synthetic_text (str): The transformed email text
        transformation_settings (dict): Settings used for transformation
    
    Returns:
        str: HTML-formatted text with color highlights
    """
    # Prepare the synthetic text for highlighting
    highlighted_text = synthetic_text
    
    # Define patterns to highlight with their corresponding colors
    highlight_patterns = [
        # Company name replacements (green)
        (fr'\b{transformation_settings["new_company"]}\b', 
         lambda m: f'<span style="background-color: #c8e6c9; padding: 2px; border-radius: 3px;">{m.group(0)}</span>'),
        
        # Email domain replacements (green)
        (fr'@{transformation_settings["email_domain"]}', 
         lambda m: f'<span style="background-color: #c8e6c9; padding: 2px; border-radius: 3px;">{m.group(0)}</span>'),
        
        # Currency values (blue)
        (r'([₹£$A\$C\$S\$R][\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh))?)',
         lambda m: f'<span style="background-color: #bbdefb; padding: 2px; border-radius: 3px;">{m.group(0)}</span>'),
        
        # Dates referencing current year (purple)
        (fr'{transformation_settings["year"]}', 
         lambda m: f'<span style="background-color: #e1bee7; padding: 2px; border-radius: 3px;">{m.group(0)}</span>'),
        
        # Phone numbers (orange)
        (r'(\+\d{1,3}[\s-]\d{1,4}[\s-]\d{3,4}[\s-]\d{3,4})',
         lambda m: f'<span style="background-color: #ffe0b2; padding: 2px; border-radius: 3px;">{m.group(0)}</span>'),
    ]
    
    # Apply highlighting patterns
    for pattern, replacer in highlight_patterns:
        highlighted_text = re.sub(pattern, replacer, highlighted_text)
    
    return highlighted_text

# === UI FUNCTIONS ===
def load_data(file):
    """Load and process uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Clean up data
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
                
        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"

def get_email_identifier(df):
    """Get the best column to identify emails"""
    priority_cols = ['subject', 'Subject', 'title', 'Title', 'id', 'ID']
    for col in priority_cols:
        if col in df.columns:
            return col
    return df.columns[0]  # Default to first column

def get_email_content_column(df):
    """Get the column containing email content"""
    content_cols = ['content', 'Content', 'body', 'Body', 'text', 'Text', 'message', 'Message']
    for col in content_cols:
        if col in df.columns:
            return col
    
    # If no obvious content column, use the column with the longest text
    max_length = 0
    content_col = df.columns[0]
    
    for col in df.columns:
        if df[col].dtype == object:  # Only check text columns
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > max_length:
                max_length = avg_length
                content_col = col
    
    return content_col

def process_emails(selected_emails, df, id_col, api_key, transformation_settings):
    """Process selected emails through the synthesis pipeline"""
    replacer = EntityReplacer(transformation_settings)
    paraphraser = LLMParaphraser(api_key)
    validator = Validator()
    
    # Find content column
    content_col = get_email_content_column(df)
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, email_id in enumerate(selected_emails):
        try:
            # Get the selected row
            email_row = df[df[id_col] == email_id].iloc[0]
            original_text = email_row[content_col]
            
            # Apply entity replacement for initial transformations
            with st.spinner(f"Applying entity replacement to email {idx+1}/{len(selected_emails)}..."):
                cleaned = replacer.replace(original_text)
            
            # Use LLM for comprehensive transformation
            with st.spinner(f"Generating synthetic version of email {idx+1}/{len(selected_emails)}..."):
                synthetic = paraphraser.paraphrase(cleaned, transformation_settings)
            
            # Validate the result
            issues = validator.validate(synthetic, transformation_settings.get("original_company", "Enron"))
            
            # Highlight changes in synthetic email
            highlighted_synthetic = highlight_changes(original_text, synthetic, transformation_settings)
            
            results.append({
                "Original Subject": email_id,
                "Original Email": original_text,
                "Synthetic Email": synthetic,
                "Highlighted Synthetic": highlighted_synthetic,
                "Validation Issues": ", ".join(issues) if issues else "None"
            })
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(selected_emails))
        
        except Exception as e:
            results.append({
                "Original Subject": email_id,
                "Original Email": f"Error accessing original content: {str(e)}",
                "Synthetic Email": f"Error: {str(e)}",
                "Highlighted Synthetic": f"Error: {str(e)}",
                "Validation Issues": "Processing failed"
            })
            
    return results

def process_direct_input(text, api_key, transformation_settings):
    """Process a directly input email text"""
    replacer = EntityReplacer(transformation_settings)
    paraphraser = LLMParaphraser(api_key)
    validator = Validator()
    
    try:
        # Apply entity replacement for initial transformations
        with st.spinner("Applying entity replacement..."):
            cleaned = replacer.replace(text)
        
        # Use LLM for comprehensive transformation
        with st.spinner("Generating synthetic version..."):
            synthetic = paraphraser.paraphrase(cleaned, transformation_settings)
        
        # Validate the result
        issues = validator.validate(synthetic, transformation_settings.get("original_company", "Enron"))
        
        # Highlight changes in synthetic email
        highlighted_synthetic = highlight_changes(text, synthetic, transformation_settings)
        
        return {
            "Original Email": text,
            "Synthetic Email": synthetic,
            "Highlighted Synthetic": highlighted_synthetic,
            "Validation Issues": ", ".join(issues) if issues else "None"
        }
    
    except Exception as e:
        return {
            "Original Email": text,
            "Synthetic Email": f"Error: {str(e)}",
            "Highlighted Synthetic": f"Error: {str(e)}",
            "Validation Issues": "Processing failed"
        }

# === MAIN APP ===
def main():
    st.title("Email Synthesizer")
    st.subheader("Generate Synthetic Emails from Enron Dataset or Custom Email Input")
    
    # === SIDEBAR CONFIGURATION ===
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # Transformation settings
    with st.sidebar.expander("Transformation Settings", expanded=False):
        # Original company name (default is Enron)
        original_company = st.text_input("Original Company Name", value="Enron")
        
        # Target company and domain
        target_company = st.text_input("Target Company Name", value="Agriculture India")
        email_domain = st.text_input("Email Domain", value="agriindia.com",
                                     help="Domain for email addresses (without @)")
        
        # Target industry
        target_industry = st.selectbox(
            "Target Industry", 
            options=["agriculture", "technology", "healthcare", "finance", "education", "manufacturing"]
        )
        
        # Target location
        target_location = st.selectbox(
            "Target Location", 
            options=["India", "UK", "Australia", "Canada", "Singapore", "South Africa"]
        )
        
        # Currency conversion rate
        default_rate = 80.0 if target_location == "India" else (
            0.75 if target_location == "UK" else 
            1.3 if target_location == "Australia" else
            1.25 if target_location == "Canada" else
            1.35 if target_location == "Singapore" else
            18.0 if target_location == "South Africa" else 1.0
        )
        
        currency_rate = st.number_input(
            "Currency Conversion Rate (USD to target)", 
            min_value=0.1, 
            max_value=100.0, 
            value=default_rate,
            help="For USD to INR use ~80, for USD to GBP use ~0.75"
        )
        
        # Target year
        target_year = st.number_input("Target Year", min_value=2023, max_value=2030, value=2024)
    
    # Collect all transformation settings in a dictionary
    transformation_settings = {
        "original_company": original_company,
        "new_company": target_company,
        "industry": target_industry,
        "year": target_year,
        "location": target_location,
        "currency_conversion_rate": currency_rate,
        "email_domain": email_domain
    }
    
    # === INPUT OPTIONS ===
    # Create tabs for different input methods
    input_tab, file_tab = st.tabs(["Direct Email Input", "Upload File"])
    
    # Tab 1: Direct Email Input
    with input_tab:
        st.subheader("Input Email Text Directly")
        
        # Text area for direct input
        direct_input = st.text_area(
            "Enter email text to transform", 
            height=200,
            placeholder="Paste your email content here..."
        )
        
        # Process direct input
        if st.button("Generate Synthetic Email", key="direct_button"):
            if not api_key:
                st.error("Please enter OpenAI API key")
            elif not direct_input or len(direct_input.strip()) < 10:
                st.error("Please enter valid email content (at least 10 characters)")
            else:
                # Process the input
                result = process_direct_input(direct_input, api_key, transformation_settings)
                
                # Display side-by-side comparison
                st.subheader("Side-by-Side Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Original Email")
                    st.text_area("", result["Original Email"], height=400, key="original_direct", disabled=True)
                
                with col2:
                    st.write("### Synthetic Email")
                    # Display highlighted synthetic email
                    st.markdown(result["Highlighted Synthetic"], unsafe_allow_html=True)
                
                st.write(f"**Validation:** {result['Validation Issues']}")
                
                show_raw_direct = st.checkbox("Show Raw Synthetic Email (for copying)")
                if show_raw_direct:
                    st.text_area("Raw Synthetic Email", result["Synthetic Email"], height=200)    
    # Tab 2: File Upload
    with file_tab:
        st.subheader("Upload Email Data File")
        
        # File upload section
        uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        
        if uploaded_file:
            df, error = load_data(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} emails")
                st.write("Sample data:")
                st.dataframe(df.head())
        
        # Email selection
        if st.session_state.df is not None:
            id_col = get_email_identifier(st.session_state.df)
            email_options = st.session_state.df[id_col].tolist()
            
            selected_emails = st.multiselect(
                "Select emails to synthesize", 
                options=email_options,
                help="Select one or more emails to transform"
            )
            
            num_to_process = len(selected_emails)
            if num_to_process > 0:
                st.info(f"Selected {num_to_process} email(s) to process")
            
            if st.button("Generate Synthetic Emails", key="file_button"):
                if not api_key:
                    st.error("Please enter OpenAI API key")
                elif not selected_emails:
                    st.error("Please select emails to process")
                else:
                    # Process the selected emails
                    st.session_state.results = process_emails(
                        selected_emails, 
                        st.session_state.df, 
                        id_col, 
                        api_key,
                        transformation_settings
                    )
        
        # Display results for file upload
        if "results" in st.session_state and st.session_state.results:
            st.subheader("Results")
            
            # Create a color legend for highlights
            with st.expander("Color Highlighting Legend", expanded=False):
                st.markdown("""
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="background-color: #c8e6c9; padding: 2px 5px; border-radius: 3px;">Company name and email domains</span>
                    </div>
                    <div>
                        <span style="background-color: #bbdefb; padding: 2px 5px; border-radius: 3px;">Currency values</span>
                    </div>
                    <div>
                        <span style="background-color: #e1bee7; padding: 2px 5px; border-radius: 3px;">Dates (year)</span>
                    </div>
                    <div>
                        <span style="background-color: #ffe0b2; padding: 2px 5px; border-radius: 3px;">Phone numbers</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            for result in st.session_state.results:
                with st.expander(f"Subject: {result['Original Subject']}"):
                    # Side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Original Email")
                        st.text_area("", result["Original Email"], height=400, key=f"original_{result['Original Subject']}", disabled=True)
                    
                    with col2:
                        st.write("### Synthetic Email")
                        # Display highlighted synthetic email
                        st.markdown(result["Highlighted Synthetic"], unsafe_allow_html=True)
                    
                    st.write(f"**Validation:** {result['Validation Issues']}")
                    
                    # Replace nested expander with checkbox
                    show_raw = st.checkbox("Show Raw Synthetic Email (for copying)", key=f"show_raw_{result['Original Subject']}")
                    if show_raw:
                        st.text_area("Raw Synthetic Email", result["Synthetic Email"], height=200, key=f"raw_synthetic_{result['Original Subject']}")
    
    # Add export functionality
    if "results" in st.session_state and st.session_state.results:
        st.subheader("Export Results")
        
        if st.button("Export as CSV"):
            # Convert results to DataFrame
            export_data = []
            for result in st.session_state.results:
                export_data.append({
                    "Original Subject": result["Original Subject"],
                    "Original Email": result["Original Email"],
                    "Synthetic Email": result["Synthetic Email"],
                    "Validation Issues": result["Validation Issues"]
                })
            
            export_df = pd.DataFrame(export_data)
            
            # Convert to CSV
            csv = export_df.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="synthetic_emails.csv",
                mime="text/csv"
            )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Email Synthesizer v1.0")
    
    # Add usage instructions
    with st.sidebar.expander("Usage Instructions", expanded=False):
        st.markdown("""
        **How to use this tool:**
        
        1. Enter your OpenAI API key in the sidebar
        2. Customize transformation settings if needed
        3. Either:
           - Upload a CSV/Excel file with email data, or
           - Paste email text directly in the input area
        4. Process the emails
        5. Review side-by-side comparisons with highlighted changes
        6. Export results if needed
        
        **What gets transformed:**
        - Company names
        - Email addresses
        - Currency values
        - Dates
        - Phone numbers
        - Names of people
        - Industry-specific terms
        """)

if __name__ == "__main__":
    main()