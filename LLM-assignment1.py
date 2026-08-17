import hashlib
import re


class CustomWordTokenizer:
    """A dynamic word tokenizer that maintains an internal vocabulary and assigns

    unique, non-sequential IDs using a deterministic hash-based allocation scheme with
    collision resolution.
    """

    def __init__(self, base_offset: int = 1000, hash_range: int = 8999):
        # 1. Reserved Special Tokens for industrial NLP standards
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.base_offset = base_offset
        self.hash_range = hash_range

    def _generate_hash_id(self, word: str) -> int:
        """Generates a non-sequential, deterministic ID using MD5 hashing

        and linear probing for collision resolution.
        """
        # Generate a 32-bit deterministic integer hash from the word string
        hash_digest = hashlib.md5(word.encode("utf-8")).digest()
        hash_int = int.from_bytes(hash_digest[:4], byteorder="big")

        # Compute candidate ID within the range [base_offset, base_offset + hash_range)
        candidate_id = self.base_offset + (hash_int % self.hash_range)

        # Linear probing to handle potential hash collisions
        while candidate_id in self.id2word:
            if self.id2word[candidate_id] == word:
                return candidate_id
            candidate_id += 1

        return candidate_id

    def tokenize(self, text: str) -> list[str]:
        """Converts raw input text into lowercase word tokens, ignoring punctuation."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_token(self, word: str) -> int:
        """Adds a word token to the internal vocabulary if unseen and returns its unique ID."""
        word = word.lower()
        if word in self.word2id:
            return self.word2id[word]

        # Generate unique hash-based ID
        new_id = self._generate_hash_id(word)

        # Update dynamic internal vocabulary
        self.word2id[word] = new_id
        self.id2word[new_id] = word
        return new_id

    def encode(self, text: str) -> tuple[list[str], list[int]]:
        """Processes input text, updates the vocabulary dynamically, and returns tokens

        along with their corresponding IDs.
        """
        tokens = self.tokenize(text)
        token_ids = [self.add_token(token) for token in tokens]
        return tokens, token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Converts a list of numerical token IDs back into a text string."""
        return " ".join([self.id2word.get(tid, "<UNK>") for tid in token_ids])

    def get_vocabulary(self) -> dict[str, int]:
        """Returns the current state of the internal word-to-ID vocabulary."""
        return self.word2id


# ==============================================================================
# DEMONSTRATION & TESTING
# ==============================================================================
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = CustomWordTokenizer()

    # Input text example
    input_text = "This is a test. This test is simple."

    # Process and encode text
    tokens, token_ids = tokenizer.encode(input_text)

    print("=" * 70)
    print("                      CUSTOM TOKENIZER OUTPUT                         ")
    print("=" * 70)
    print(f"\n1. Raw Input Text:\n   '{input_text}'")
    print(f"\n2. Extracted Tokens:\n   {tokens}")
    print(f"\n3. Numerical Token IDs:\n   {token_ids}")

    print("\n4. Dynamically Maintained Internal Vocabulary Map:")
    # Exclude reserved special tokens for display clarity
    regular_vocab = {
        k: v for k, v in tokenizer.get_vocabulary().items() if v >= 1000
    }
    print("  ", regular_vocab)

    print("\n5. Decoding Token IDs back to text:")
    decoded_text = tokenizer.decode(token_ids)
    print(f"   '{decoded_text}'")

    # Processing a new sentence to demonstrate vocabulary expansion & reuse
    print("\n" + "-" * 70)
    new_text = "This simple test is new!"
    print(f"6. Processing New Text:\n   '{new_text}'")
    new_tokens, new_token_ids = tokenizer.encode(new_text)
    print(f"   New Token IDs: {new_token_ids}")
    print("\n   Updated Vocabulary Map:")
    regular_vocab_updated = {
        k: v for k, v in tokenizer.get_vocabulary().items() if v >= 1000
    }
    print("  ", regular_vocab_updated)