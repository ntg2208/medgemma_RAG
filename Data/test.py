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
        max_tokens=64000,
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

def fix_typos_with_llm(content, system_prompt):
    """Use LLM to fix OCR typos while preserving medical terminology.

    For large documents (>32k chars), chunks into 20k segments for processing.
    """
    CHUNK_SIZE = 20000
    MAX_SIZE = 32000

    # If content is small enough, process in one go
    if len(content) <= MAX_SIZE:
        prompt = f"""Please review the following medical document and fix OCR errors:
1. Add spaces between words that are incorrectly joined (e.g., "kidneydisease" → "kidney disease")
2. Fix obvious typos and misspellings
3. IMPORTANT: Preserve all medical terminology exactly as intended
4. Keep the original markdown formatting
5. Do NOT add explanations or comments

Document:
{content}

Return ONLY the corrected text."""

        return response(system_prompt, prompt)

    # For large documents, chunk and process
    print(f"   Document is {len(content)} chars, chunking into {CHUNK_SIZE} char segments...")

    chunks = []
    start = 0
    chunk_num = 0

    while start < len(content):
        end = start + CHUNK_SIZE

        # Try to break at a paragraph boundary
        if end < len(content):
            # Look for the last paragraph break within the chunk
            paragraph_break = content.rfind('\n\n', start, end)
            if paragraph_break > start:
                end = paragraph_break + 2  # Include the newlines

        chunk = content[start:end]
        chunks.append((chunk_num, chunk))
        chunk_num += 1
        start = end

    print(f"   Split into {len(chunks)} chunks")

    # Process each chunk
    fixed_chunks = []
    for i, (chunk_id, chunk) in enumerate(chunks, 1):
        print(f"   Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)...")

        prompt = f"""Please review the following medical document segment and fix OCR errors:
1. Add spaces between words that are incorrectly joined (e.g., "kidneydisease" → "kidney disease")
2. Fix obvious typos and misspellings
3. IMPORTANT: Preserve all medical terminology exactly as intended
4. Keep the original markdown formatting
5. Do NOT add explanations or comments
6. This is part {i} of {len(chunks)} - preserve continuity

Document segment:
{chunk}

Return ONLY the corrected text."""

        fixed_chunk = response(system_prompt, prompt)
        fixed_chunks.append(fixed_chunk)

    # Combine chunks
    fixed_content = ''.join(fixed_chunks)
    print(f"   ✓ Combined {len(chunks)} chunks into {len(fixed_content)} chars")

    return fixed_content

def extract_metadata_with_llm(content, system_prompt):
    """Extract structured metadata from the document."""
    prompt = f"""Analyze this medical document and extract:
1. Recommended title (concise, descriptive)
2. Document type (guideline, patient info, clinical protocol, etc.)
3. Main topic areas (list)
4. Target audience (patients, healthcare providers, specialists)

Document:
{content}

Return as JSON format."""

    return response(system_prompt, prompt)

content = clean_content(content)

# %%
def process_document(filename, input_dir="processed_ocr"):
    """Complete document processing pipeline."""
    import json

    print(f"Processing: {filename}")
    print("=" * 80)

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

    # Fix typos and OCR errors with LLM
    print("\n🔧 Fixing typos and OCR errors...")
    fixed_content = fix_typos_with_llm(cleaned_content, system_prompt)
    print(f"✅ Fixed: {len(fixed_content)} characters")

    # Generate title (ONLY the title, nothing else)
    print("\n📝 Generating title...")
    title_prompt = f"""Read this medical document and provide a concise, descriptive title.

Document:
{fixed_content}

Output format: Return ONLY the title text, no quotes, no explanation, no additional text."""

    title = response(system_prompt, title_prompt).strip()
    # Remove any quotes or extra formatting
    title = title.strip('"\'').strip()
    # Take only the first line if multiple lines returned
    title = title.split('\n')[0].strip()
    print(f"✅ Title: {title}")

    # Generate summary
    print("\n📋 Generating summary...")
    summary_prompt = f"""Summarize this medical document in 2-3 sentences.

Document:
{fixed_content}"""
    summary = response(system_prompt, summary_prompt).strip()
    print(f"✅ Summary: {summary[:200]}...")

    # Save cleaned markdown file
    cleaned_md_file = os.path.join(input_dir, filename.replace('.md', '_cleaned.md'))
    with open(cleaned_md_file, 'w') as f:
        f.write(fixed_content)
    print(f"\n💾 Saved cleaned content: {cleaned_md_file}")

    # Save metadata JSON (without content)
    metadata_file = os.path.join(input_dir, filename.replace('.md', '_metadata.json'))
    metadata = {
        "original_file": filename,
        "cleaned_file": filename.replace('.md', '_cleaned.md'),
        "title": title,
        "summary": summary,
        "stats": {
            "original_length": len(raw_content),
            "cleaned_length": len(cleaned_content),
            "fixed_length": len(fixed_content),
        }
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_file}")

    print("=" * 80)

    return metadata

# %% Batch processing
def batch_process_documents(input_dir="processed_ocr"):
    """Process all markdown files in a directory."""
    import glob

    # Find all .md files that are not already cleaned
    md_files = [f for f in glob.glob(os.path.join(input_dir, "*.md"))
                if not f.endswith('_cleaned.md')]
    print(f"Found {len(md_files)} markdown files to process")

    results = []
    for i, filepath in enumerate(md_files, 1):
        filename = os.path.basename(filepath)
        print(f"\n[{i}/{len(md_files)}] Processing {filename}...")

        try:
            result = process_document(filename, input_dir)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Processed {len(results)}/{len(md_files)} documents successfully")
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
print("\n🚀 Batch processing all documents...\n")
batch_results = batch_process_documents()

# %%
