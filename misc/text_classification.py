import pandas as pd
import numpy as np
import anthropic
import re
import json
import ast
from collections import Counter
from tqdm import tqdm
import time
import os


class ClaudeAPIClassifier:
    def __init__(self, api_key=None):
        """
        Direct LLM-based label extraction using Anthropic Claude API
        """
        if not api_key:
            raise ValueError("Please provide your Anthropic API key")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Rate limiting settings
        self.requests_per_minute = 50  # Adjust based on your API limits
        self.last_request_time = 0
    
    def rate_limit(self):
        """Simple rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def create_extraction_prompt(self, text, label_type="keywords"):
        """Create the extraction prompt for Claude API"""
        
        # Clean the text first
        clean_text = self.clean_text_for_extraction(text)
        
        if label_type == "keywords":
            prompt = f"""Extract the top-3 most important keywords from the following historical text. Return ONLY a Python list format.

Text: "{clean_text}"

Return format: ['keyword1', 'keyword2', 'keyword3']"""

        elif label_type == "categories":
            prompt = f"""Extract the top-3 most specific historical categories that describe this text. Focus on specific historical contexts, locations, or events. Return ONLY a Python list format.

Text: "{clean_text}"

Return format: ['category1', 'category2', 'category3']"""

        elif label_type == "clip_terms":
            prompt = f"""Extract the top-3 best image search terms that would help find a photograph of this scene using CLIP. Focus on visual elements, objects, and scenes. Return ONLY a Python list format.

Text: "{clean_text}"

Return format: ['visual_term1', 'visual_term2', 'visual_term3']"""

        elif label_type == "entities":
            prompt = f"""Extract the top-3 most important named entities (people, places, organizations, specific locations) from this text. Return ONLY a Python list format.

Text: "{clean_text}"

Return format: ['entity1', 'entity2', 'entity3']"""

        return prompt
    
    def clean_text_for_extraction(self, text):
        """Clean text for better Claude extraction"""
        if pd.isna(text):
            return ""
        
        # Remove brackets but keep content
        text = re.sub(r'^\[(.*?)\]\s*', r'\1. ', text)
        
        # Remove common metadata phrases
        text = re.sub(r'\baccording to [^.]*\.?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bimage of\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bphotograph\b', '', text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for API efficiency
        words = text.split()
        if len(words) > 150:
            text = ' '.join(words[:150]) + "..."
        
        return text
    
    def extract_labels_with_claude(self, text, label_types=None):
        """Extract labels using Claude API"""
        if label_types is None:
            label_types = ["keywords", "categories", "clip_terms", "entities"]
        
        results = {}
        
        for label_type in label_types:
            try:
                # Rate limiting
                self.rate_limit()
                
                # Create prompt
                prompt = self.create_extraction_prompt(text, label_type)
                
                # Call Claude API
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast and cost-effective
                    max_tokens=100,
                    temperature=0.1,  # Low temperature for consistent output
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extract the response text
                response_text = response.content[0].text.strip()
                
                # Parse the response to extract the list
                labels = self.parse_claude_response(response_text)
                results[label_type] = labels
                
            except Exception as e:
                print(f"Error extracting {label_type}: {e}")
                results[label_type] = ["api_error"]
                # Add a longer delay after errors
                time.sleep(2)
        
        return results
    
    def parse_claude_response(self, response_text):
        """Parse Claude response to extract the Python list"""
        try:
            # Claude should return clean lists, but let's be robust
            
            # First, try to find a Python list pattern
            list_patterns = [
                r"\[(.*?)\]",  # Standard list format
                r"'([^']+)',?\s*'([^']+)',?\s*'([^']+)'",  # Three quoted items
            ]
            
            for pattern in list_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    if len(match.groups()) == 1:
                        # Single group - full list content
                        list_content = match.group(1)
                        try:
                            # Try to evaluate as Python list
                            labels = ast.literal_eval(f'[{list_content}]')
                            # Ensure all are strings and clean them
                            clean_labels = []
                            for label in labels:
                                clean_label = str(label).strip().strip('"\'')
                                if clean_label and len(clean_label) > 1:
                                    clean_labels.append(clean_label)
                            return clean_labels[:3]  # Top 3
                        except:
                            # Fallback: split by commas and clean
                            items = [item.strip().strip('"\'') for item in list_content.split(',')]
                            return [item for item in items if item and len(item) > 1][:3]
                    else:
                        # Multiple groups - individual captures
                        labels = [group.strip().strip('"\'') for group in match.groups() if group and group.strip()]
                        return labels[:3]
            
            # If no list pattern found, try to extract quoted items
            quoted_items = re.findall(r"'([^']+)'|\"([^\"]+)\"", response_text)
            if quoted_items:
                labels = [item[0] or item[1] for item in quoted_items if item[0] or item[1]]
                return labels[:3]
            
            # Last resort: split by common delimiters
            if ',' in response_text:
                items = [item.strip().strip('"\'') for item in response_text.split(',')]
                valid_items = [item for item in items if item and len(item) > 1 and not item.startswith('[') and not item.endswith(']')]
                if valid_items:
                    return valid_items[:3]
            
        except Exception as e:
            print(f"Error parsing Claude response: {e}")
        
        return ["parsing_error"]
    
    def create_combined_category(self, extraction_results):
        """Create a combined category from multiple extraction types"""
        keywords = extraction_results.get('keywords', [])
        categories = extraction_results.get('categories', [])
        entities = extraction_results.get('entities', [])
        
        # Filter out errors
        keywords = [k for k in keywords if k not in ['api_error', 'parsing_error']]
        categories = [c for c in categories if c not in ['api_error', 'parsing_error']]
        entities = [e for e in entities if e not in ['api_error', 'parsing_error']]
        
        category_parts = []
        
        # Priority 1: Use the most specific category if available
        if categories:
            primary_category = categories[0].replace(' ', '_').title()
            category_parts.append(primary_category)
        
        # Priority 2: Add specific entity (location, organization)
        if entities:
            entity = entities[0].replace(' ', '_').title()
            # Only add if it's not already included in the category
            if not category_parts or entity.lower() not in category_parts[0].lower():
                category_parts.append(entity)
        
        # Priority 3: Add most relevant keyword if there's space
        if keywords and len(category_parts) < 2:
            keyword = keywords[0].replace(' ', '_').title()
            # Only add if it adds new information
            existing = ' '.join(category_parts).lower() if category_parts else ""
            if keyword.lower() not in existing:
                category_parts.append(keyword)
        
        # Create final category name
        if category_parts:
            return '_'.join(category_parts[:3])  # Max 3 parts
        elif keywords:
            return f"Historical_{keywords[0].replace(' ', '_').title()}"
        else:
            return "General_Historical_Content"
    
    def classify_dataset_with_claude(self, csv_path, output_path=None, sample_size=None, api_key=None):
        """Main classification using Claude API"""
        
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"Processing {len(df)} records with Claude API...")
        print("This may take a while due to API rate limiting...")
        
        # Define extraction types
        label_types = ["keywords", "categories", "clip_terms", "entities"]
        
        # Process each record
        results = []
        all_extractions = {label_type: [] for label_type in label_types}
        category_counter = Counter()
        
        # Add progress tracking
        successful_extractions = 0
        failed_extractions = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Claude API extraction"):
            text = row['enriched_document_description']
            
            try:
                # Extract labels using Claude
                extraction_results = self.extract_labels_with_claude(text, label_types)
                
                # Create combined category
                combined_category = self.create_combined_category(extraction_results)
                
                # Create result record
                result = {
                    'img_url': row['img_url'],
                    'original_description': text,
                    'cleaned_text': self.clean_text_for_extraction(text),
                    'combined_category': combined_category
                }
                
                # Add individual extraction results
                for label_type in label_types:
                    labels = extraction_results.get(label_type, [])
                    result[f'{label_type}_extracted'] = labels
                    result[f'{label_type}_primary'] = labels[0] if labels else "none"
                    
                    # Collect for statistics
                    all_extractions[label_type].extend(labels)
                
                results.append(result)
                category_counter[combined_category] += 1
                successful_extractions += 1
                
            except Exception as e:
                print(f"Failed to process row {idx}: {e}")
                failed_extractions += 1
                continue
        
        print(f"\nProcessing complete: {successful_extractions} successful, {failed_extractions} failed")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Display results
        self.display_claude_results(category_counter, all_extractions, results_df)
        
        # Generate CLIP mappings
        clip_mappings = self.generate_clip_mappings_claude(results_df)
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            
            # Save CLIP mappings
            clip_output = output_path.replace('.csv', '_clip_mappings.json')
            with open(clip_output, 'w') as f:
                json.dump(clip_mappings, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
            print(f"CLIP mappings saved to {clip_output}")
        
        return results_df, clip_mappings
    
    def display_claude_results(self, category_counter, all_extractions, results_df):
        """Display Claude API extraction results"""
        print("\n" + "="*80)
        print("CLAUDE API LABEL EXTRACTION RESULTS")
        print("="*80)
        
        # Show combined categories
        print(f"\n📋 COMBINED CATEGORIES:")
        print("-" * 60)
        total = sum(category_counter.values())
        for category, count in category_counter.most_common(20):
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{category}: {count} ({percentage:.1f}%)")
        
        # Show extraction statistics
        for label_type, extractions in all_extractions.items():
            print(f"\n📊 {label_type.upper()} EXTRACTIONS:")
            print("-" * 40)
            
            # Remove errors and empty
            valid_extractions = [e for e in extractions if e not in ['api_error', 'parsing_error', 'none', '']]
            
            if valid_extractions:
                extraction_counts = Counter(valid_extractions)
                for item, count in extraction_counts.most_common(15):
                    print(f"  {item}: {count}")
            else:
                print("  No valid extractions")
        
        # Show sample results
        print("\n" + "="*80)
        print("SAMPLE CLAUDE CLASSIFICATIONS:")
        print("="*80)
        
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            print(f"\nText: {row['cleaned_text'][:100]}...")
            print(f"Combined Category: {row['combined_category']}")
            print(f"Keywords: {row['keywords_extracted']}")
            print(f"Categories: {row['categories_extracted']}")
            print(f"CLIP Terms: {row['clip_terms_extracted']}")
            print(f"Entities: {row['entities_extracted']}")
    
    def generate_clip_mappings_claude(self, results_df):
        """Generate CLIP mappings from Claude extractions"""
        clip_mappings = {}
        
        for category in results_df['combined_category'].unique():
            if category in ["General_Historical_Content"]:
                continue
            
            # Get examples of this category
            examples = results_df[results_df['combined_category'] == category]
            
            # Collect CLIP terms from examples
            clip_terms = []
            keywords = []
            
            for _, example in examples.head(3).iterrows():
                # Get CLIP terms
                terms = example.get('clip_terms_extracted', [])
                if isinstance(terms, list):
                    clip_terms.extend([t for t in terms if t not in ['api_error', 'parsing_error']])
                
                # Get keywords as fallback
                kw = example.get('keywords_extracted', [])
                if isinstance(kw, list):
                    keywords.extend([k for k in kw if k not in ['api_error', 'parsing_error']])
            
            # Create query from most relevant terms
            if clip_terms:
                # Use CLIP-specific terms (prioritized)
                term_counts = Counter(clip_terms)
                top_terms = [term for term, count in term_counts.most_common(4)]
                query = ' '.join(top_terms).lower()
            elif keywords:
                # Fallback to keywords
                keyword_counts = Counter(keywords)
                top_keywords = [kw for kw, count in keyword_counts.most_common(3)]
                query = ' '.join(top_keywords).lower()
            else:
                # Final fallback to category name
                query = category.replace('_', ' ').lower()
            
            clip_mappings[category] = query
        
        return clip_mappings

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    # Initialize Claude classifier
    classifier = ClaudeAPIClassifier(api_key=api_key)
    
    csv_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv"
    output_path = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/claude_api_results.csv"
    
    # Run Claude API classification
    results_df, clip_mappings = classifier.classify_dataset_with_claude(
        csv_path, 
        output_path,
        sample_size=100,  # Start with 100 to test
        api_key=api_key
    )
    
    print(f"\nGenerated {len(set(results_df['combined_category']))} unique categories")
    
    # Show CLIP mappings
    print("\n" + "="*60)
    print("CLAUDE API CLIP QUERY MAPPINGS:")
    print("="*60)
    
    for category, query in list(clip_mappings.items())[:10]:
        print(f"{category}: '{query}'")
    
    return results_df, clip_mappings

if __name__ == "__main__":
    results, mappings = main()