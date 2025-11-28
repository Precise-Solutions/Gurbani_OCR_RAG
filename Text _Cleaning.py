import regex as re

def clean_gurmukhi_text(text):
    # Remove extra spaces between Gurmukhi diacritics and letters
    text = re.sub(r'(\p{L})\s+(\p{Mn})', r'\1\2', text)

    # Remove spaces inside a Gurmukhi word
    text = re.sub(r'(?<=\p{L})\s+(?=\p{L})', '', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)

    # Fix issues with dangling matras
    text = re.sub(r'\s([਼ੑੂੁੰਂਿੀੇੈੋੌਾ੍])+', r'\1', text)

    # Trim leading/trailing spaces
    return text.strip()


# Example usage:
with open("gurbani.txt", "r") as f:
    raw = f.read()

cleaned = clean_gurmukhi_text(raw)

with open("gurbani_cleaned.txt", "w") as f:
    f.write(cleaned)

print("Done!")
