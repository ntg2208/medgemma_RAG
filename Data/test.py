# %%
import logging
import os
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

data_dir = "processed_ocr"

# %%
def response(system_prompt, prompt, history=None, stream_output=False):
    """
    Send a prompt to vLLM and get the response.

    Args:
        system_prompt (str): System prompt to set context/behavior
        prompt (str): User prompt/query
        history (list, optional): List of message dicts with 'role' and 'content' keys
                                  Example: [{"role": "user", "content": "..."},
                                           {"role": "assistant", "content": "..."}]
        stream_output (bool): If True, print response as it streams. Default False.

    Returns:
        str: Complete response from the model
    """
    # Get server URL from environment or use localhost
    server_url = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")

    # Initialize OpenAI client pointing to vLLM
    client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=f"{server_url}/v1"
    )

    # Build messages list
    messages = []

    # Add system prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add history if provided
    if history:
        messages.extend(history)

    # Add current user prompt
    messages.append({"role": "user", "content": prompt})

    # Stream the response
    stream = client.chat.completions.create(
        model="google/medgemma-1.5-4b-it",
        messages=messages,
        stream=True,
        max_tokens=32000,  # Sufficient for titles and summaries
        temperature=0.7,
    )

    # Collect and optionally print streamed chunks
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            if stream_output:
                print(content, end="", flush=True)
            full_response += content

    if stream_output:
        print()  # New line after streaming
    return full_response


# %%
# Test vLLM streaming on port 8000
def test_vllm_streaming():
    """Test streaming from vLLM server running on port 8000"""

    server_url = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")

    print(f"🔗 Connecting to vLLM at {server_url}")
    print("=" * 80)

    try:
        # Test 1: Simple query without history
        print("\n📝 Test 1: Simple query")
        print("-" * 80)
        system_prompt = "You are a helpful medical assistant specializing in chronic kidney disease."
        prompt = "What is chronic kidney disease?"

        response_text = response(system_prompt, prompt, stream_output=True)
        print("-" * 80)
        print(f"✅ Received {len(response_text)} characters\n")

        # Test 2: Query with conversation history
        print("\n📝 Test 2: Query with history")
        print("-" * 80)
        history = [
            {"role": "user", "content": "What is CKD?"},
            {"role": "assistant", "content": "CKD stands for Chronic Kidney Disease."}
        ]
        prompt2 = "What are the stages?"

        response_text2 = response(system_prompt, prompt2, history, stream_output=True)
        print("-" * 80)
        print(f"✅ Received {len(response_text2)} characters\n")

        print("=" * 80)
        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure vLLM is running on port 8000:")
        print("   ./scripts/start-model-server.sh")

# %%
filename = os.path.join(data_dir, "85089pfood.md")

with open(filename, "r") as f:
    content = f.read()
    
print(content[:500])  # Print the first 500 characters of the file

def clean_content(content):
    """Clean the content by removing artifacts from OCR processing."""
    import re

    text = content

    # Remove image placeholders
    text = text.replace("<!-- image -->", "")

    # Remove dash bullet markers at start of lines
    text = re.sub(r'^\s*-\s+', '', text, flags=re.MULTILINE)

    # Remove repetitive dots (3 or more consecutive dots)
    text = re.sub(r'\.{3,}', '', text)

    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove extra spaces (but preserve single spaces)
    text = re.sub(r' {2,}', ' ', text)

    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Remove empty lines at start and end
    text = text.strip()

    return text


content = clean_content(content)

# %%
def process_document(filename, input_dir="processed_ocr", output_dir="cleaned_documents"):
    """Complete document processing pipeline."""
    import json
    import re

    print(f"Processing: {filename}")
    print("=" * 80)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load document
    filepath = os.path.join(input_dir, filename)
    with open(filepath, "r") as f:
        raw_content = f.read()

    print(f"📄 Loaded document: {len(raw_content)} characters")

    # Clean content (basic cleanup)
    cleaned_content = clean_content(raw_content)
    print(f"🧹 Basic cleaning: {len(cleaned_content)} characters")

    # Initialize LLM
    system_prompt = "You are a helpful medical assistant specializing in chronic kidney disease."

    # Generate title (ONLY the title, nothing else)
    print("\n📝 Generating title...")
    title_prompt = f"""Read this medical document and provide a concise, descriptive title.

IMPORTANT: Return ONLY the title text itself. Do not include:
- Quotation marks
- "Title:" or similar labels
- Any explanatory text
- Multiple lines
- Markdown formatting (no ##, no **, no bullets)

Examples of correct output:
Dietary Guidelines for Chronic Kidney Disease Management
Managing Diabetes in CKD Patients
Phosphate Binders and Cardiovascular Risk
Nutritional Recommendations for Stage 3-5 CKD
KDIGO Clinical Practice Guideline for Anemia in CKD
Conservative Management Options for End-Stage Renal Disease

Examples of INCORRECT output:
## Dietary Guidelines for CKD
**Managing Diabetes in CKD**
Title: Phosphate Binders
"KDIGO Guidelines"

Document:
{cleaned_content[:5000]}"""

    title = response(system_prompt, title_prompt).strip()
    # Remove any quotes or extra formatting
    title = title.strip('"\'').strip()
    # Remove any markdown symbols
    title = title.lstrip('#').strip()
    title = title.strip('*').strip()
    # Take only the first line if multiple lines returned
    title = title.split('\n')[0].strip()
    print(f"✅ Title: {title}")

    # Generate summary
    print("\n📋 Generating summary...")
    # Use only first 5000 characters for summary to reduce processing time
    content_preview = cleaned_content[:5000]
    summary_prompt = f"""Summarize this medical document in 2-3 sentences.

IMPORTANT: Return ONLY the summary text itself. Do not include:
- Introductory phrases like "Here is the summary:" or "Okay, here is..."
- Quotation marks around the summary
- Any meta-commentary about the task
- Extra line breaks or formatting

Examples of correct output:
- This guideline provides recommendations for managing chronic kidney disease in adults. It covers screening, diagnosis, and treatment options including dietary modifications and medication management. The document emphasizes early detection and intervention to slow disease progression.
- The document outlines dietary restrictions for CKD patients, focusing on sodium, potassium, and phosphorus intake. It provides specific food recommendations and meal planning strategies for different CKD stages.

Document:
{content_preview}"""
    summary = response(system_prompt, summary_prompt).strip()
    print(f"✅ Summary: {summary[:200]}...")

    # Sanitize title for filename
    safe_filename = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid characters
    safe_filename = safe_filename.replace(' ', '_')  # Replace spaces with underscores
    safe_filename = safe_filename[:100]  # Limit length to avoid filesystem issues

    # Save cleaned markdown file with title-based filename
    cleaned_md_file = os.path.join(output_dir, f"{safe_filename}.md")
    with open(cleaned_md_file, 'w') as f:
        f.write(cleaned_content)
    print(f"\n💾 Saved cleaned content: {cleaned_md_file}")

    # Save metadata JSON with title-based filename
    metadata_file = os.path.join(output_dir, f"{safe_filename}.json")
    metadata = {
        "original_file": filename,
        "title": title,
        "summary": summary,
        "cleaned_file": f"{safe_filename}.md",
        "metadata_file": f"{safe_filename}.json",
        "stats": {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned_content),
        }
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_file}")

    print("=" * 80)

    return metadata

# %% Batch processing
def batch_process_documents(input_dir="processed_ocr", output_dir="cleaned_documents", max_workers=5):
    """Process all markdown files in a directory with parallel processing.

    Args:
        input_dir: Directory containing markdown files to process
        output_dir: Directory to save cleaned files
        max_workers: Number of parallel workers (default 5)
    """
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    # Thread-safe print lock
    print_lock = Lock()

    def process_with_progress(filename, idx, total):
        """Wrapper function for thread-safe processing with progress."""
        try:
            with print_lock:
                print(f"\n[{idx}/{total}] 🚀 Starting: {filename}")

            result = process_document(filename, input_dir, output_dir)

            with print_lock:
                print(f"[{idx}/{total}] ✅ Completed: {filename}")

            return (True, result, filename)
        except Exception as e:
            with print_lock:
                print(f"[{idx}/{total}] ❌ Error processing {filename}: {e}")
            import traceback
            with print_lock:
                traceback.print_exc()
            return (False, None, filename)

    # Find all .md files that are not already cleaned
    md_files = [f for f in glob.glob(os.path.join(input_dir, "*.md"))
                if not f.endswith('_cleaned.md')]

    print(f"{'='*80}")
    print(f"Found {len(md_files)} markdown files to process")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {max_workers}")
    print(f"{'='*80}")

    if not md_files:
        print("No files to process!")
        return []

    results = []
    errors = []

    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_with_progress,
                os.path.basename(filepath),
                i,
                len(md_files)
            ): os.path.basename(filepath)
            for i, filepath in enumerate(md_files, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            success, result, filename = future.result()
            if success:
                results.append(result)
            else:
                errors.append(filename)

    print(f"\n{'='*80}")
    print(f"✅ Processed {len(results)}/{len(md_files)} documents successfully")
    if errors:
        print(f"❌ Failed: {len(errors)} documents")
        print(f"   Failed files: {', '.join(errors)}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*80}")
    return results

# %% Example: Process single document
if __name__ == "__main__":
    # Test with single document
    print("\n🚀 Processing single document...\n")
    result = process_document("85089pfood.md")
    print("\n📊 Results:")
    print(f"Title: {result['title']}")
    print(f"Summary: {result['summary']}")

# %% Example: Batch process all documents
# Uncomment to run batch processing
# Adjust max_workers based on your GPU memory and vLLM config
# Recommended: 3-5 workers for 4B model on 24GB GPU
print("\n🚀 Batch processing all documents with parallel workers...\n")
batch_results = batch_process_documents(max_workers=5)

# %%
