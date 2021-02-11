def cleanNames(text, word_length=1):
    """Clean the text feature to be consumable for the model training

    Args:
        text (str): the text feature
        word_length (int): the minimal length of a word
        
    Process:
    1. Remove all numbers
    2. Remove punctuation: `?` `!` `'` `"` `#` `:`
    3. Replace separators with spaces: `.` `,` `)` `(` `\` `/` `-`
    4. Remove all words shorter than a configurable length
    5. Replace sequences of more than one space with one space.
    6. lower the cases

    Return:
        cleaned_text (str): cleaned text
    """

    # 1. Remove all numbers
    remove_numbers = str.maketrans(dict.fromkeys("0123456789"))
    modified_text = text.translate(remove_numbers)

    # 2. Remove punctuation: `?` `!` `'` `"` `#` `:` `~` `]` `[`
    punctuation = set("!\"#':?~][")
    modified_text = "".join([i for i in modified_text if i not in punctuation])

    # 3. Replace separators with spaces: `.` `,` `)` `(` `\` `/` `-`
    remove_punctuation = str.maketrans(dict.fromkeys(".,)(\/-"))
    modified_text = modified_text.translate(remove_punctuation)
    
    # 4. Remove all words shorter than a configurable length
    # -> will set the default word length as 1
    modified_text = " ".join(
        [w for w in modified_text.split() if len(w) > word_length]
    )

    # 5. Replace sequences of more than one space with one space.
    modified_text = " ".join(modified_text.split())
    
    # 6. lower the cases
    cleaned_text = modified_text.lower()

    return cleaned_text