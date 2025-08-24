#!/usr/bin/env python3
"""
Wikipedia Content Extractor for iPad Information
This script extracts comprehensive information about iPads from Wikipedia
and saves it to separate text files for each iPad category.
"""

import wikipedia
import os
from datetime import datetime

def extract_ipad_content_separate():
    """Extract comprehensive iPad information from Wikipedia and save to separate files"""
    
    print("🔍 Extracting iPad information from Wikipedia...")
    
    # Set up Wikipedia
    wikipedia.set_lang('en')
    
    # Define iPad categories with their respective pages
    ipad_categories = {
        "ipad_general": [
            "iPad",
            "iPad (1st generation)",
            "iPad (2nd generation)",
            "iPad (3rd generation)",
            "iPad (4th generation)",
            "iPad (5th generation)",
            "iPad (6th generation)",
            "iPad (7th generation)",
            "iPad (8th generation)",
            "iPad (9th generation)",
            "iPad (10th generation)",
            "iPad (11th generation)"
        ],
        "ipad_pro": [
            "iPad Pro",
            "iPad Pro (1st generation)",
            "iPad Pro (2nd generation)",
            "iPad Pro (3rd generation)",
            "iPad Pro (4th generation)",
            "iPad Pro (5th generation)",
            "iPad Pro (6th generation)",
            "iPad Pro (7th generation)"
        ],
        "ipad_air": [
            "iPad Air",
            "iPad Air (1st generation)",
            "iPad Air (2nd generation)",
            "iPad Air (3rd generation)",
            "iPad Air (4th generation)",
            "iPad Air (5th generation)",
            "iPad Air (6th generation)",
            "iPad Air (7th generation)"
        ],
        "ipad_mini": [
            "iPad Mini",
            "iPad Mini (1st generation)",
            "iPad Mini (2nd generation)",
            "iPad Mini (3rd generation)",
            "iPad Mini (4th generation)",
            "iPad Mini (5th generation)",
            "iPad Mini (6th generation)",
            "iPad Mini (7th generation)"
        ]
    }
    
    extracted_files = []
    
    for category, pages in ipad_categories.items():
        print(f"\n📁 Processing category: {category}")
        
        category_content = []
        successful_pages = 0
        
        for page_title in pages:
            try:
                print(f"📖 Extracting: {page_title}")
                
                # Get the page
                page = wikipedia.page(page_title, auto_suggest=False)
                
                # Extract content
                content = f"# {page.title}\n\n"
                content += f"URL: {page.url}\n"
                content += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                content += page.content
                
                category_content.append(content)
                successful_pages += 1
                
                print(f"✅ Successfully extracted {page_title}")
                
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"⚠️  Disambiguation error for {page_title}: {e}")
                # Try to get the first option
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    content = f"# {page.title} (from disambiguation)\n\n"
                    content += f"URL: {page.url}\n"
                    content += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    content += page.content
                    category_content.append(content)
                    successful_pages += 1
                    print(f"✅ Successfully extracted {e.options[0]} (first option)")
                except Exception as e2:
                    print(f"❌ Failed to extract disambiguation option: {e2}")
                    
            except wikipedia.exceptions.PageError:
                print(f"❌ Page not found: {page_title}")
            except Exception as e:
                print(f"❌ Error extracting {page_title}: {e}")
        
        # Combine content for this category
        if category_content:
            combined_content = "\n\n" + "="*80 + "\n\n".join(category_content)
            
            # Save to category-specific file
            filename = f"{category}_wikipedia_content.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            
            extracted_files.append(filename)
            print(f"📄 Saved {category} content to: {filename}")
            print(f"📊 {category} file size: {len(combined_content)} characters")
            print(f"✅ Successfully extracted {successful_pages} pages for {category}")
    
    return extracted_files

def extract_specific_ipad_info_separate():
    """Extract specific iPad information with better structure to separate files"""
    
    print("🔍 Extracting structured iPad information to separate files...")
    
    extracted_files = []
    
    # Define categories and their main pages
    categories = {
        "ipad_general": "iPad",
        "ipad_pro": "iPad Pro", 
        "ipad_air": "iPad Air",
        "ipad_mini": "iPad Mini"
    }
    
    for category, main_page in categories.items():
        print(f"\n📁 Processing: {category}")
        
        try:
            # Get the main page for this category
            page = wikipedia.page(main_page, auto_suggest=False)
            
            # Create structured content
            content = f"""# {page.title} Information from Wikipedia

Source: {page.url}
Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Main Article

{page.content}

"""
            
            # Add information about specific generations if available
            if category == "ipad_general":
                # Add some general iPad models
                additional_models = ["iPad (1st generation)", "iPad (2nd generation)", "iPad (3rd generation)"]
            elif category == "ipad_pro":
                additional_models = ["iPad Pro (1st generation)", "iPad Pro (2nd generation)", "iPad Pro (3rd generation)"]
            elif category == "ipad_air":
                additional_models = ["iPad Air (1st generation)", "iPad Air (2nd generation)", "iPad Air (3rd generation)"]
            elif category == "ipad_mini":
                additional_models = ["iPad Mini (1st generation)", "iPad Mini (2nd generation)", "iPad Mini (3rd generation)"]
            else:
                additional_models = []
            
            # Add additional model information
            for model in additional_models:
                try:
                    model_page = wikipedia.page(model, auto_suggest=False)
                    content += f"\n\n## {model}\n\n"
                    content += f"Source: {model_page.url}\n\n"
                    content += model_page.content
                    print(f"✅ Added {model} information to {category}")
                except Exception as e:
                    print(f"⚠️  Could not extract {model}: {e}")
            
            # Save to category-specific file
            filename = f"{category}_wikipedia_content.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            extracted_files.append(filename)
            print(f"📄 Saved {category} content to: {filename}")
            print(f"📊 {category} file size: {len(content)} characters")
            
        except Exception as e:
            print(f"❌ Error extracting {category}: {e}")
    
    return extracted_files

if __name__ == "__main__":
    print("🚀 Wikipedia iPad Content Extractor")
    print("=" * 50)
    
    # Choose extraction method
    print("Choose extraction method:")
    print("1. Extract all iPad-related pages to separate files (comprehensive)")
    print("2. Extract main iPad categories to separate files (faster)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        files = extract_ipad_content_separate()
    elif choice == "2":
        files = extract_specific_ipad_info_separate()
    else:
        print("Invalid choice. Using method 2 (faster).")
        files = extract_specific_ipad_info_separate()
    
    if files:
        print(f"\n🎉 Successfully created {len(files)} files:")
        for file in files:
            print(f"  📄 {file}")
        print(f"\n✅ Ready to use these files with your chatbot!")
        print("Run: python chatbot.py")
