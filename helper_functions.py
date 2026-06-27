from typing import List
import re
import pandas as pd

def clean_utterance(utterance: List[str]) -> str:
    """
    Clean the utterance by removing any non-word tokens and converting to lowercase.
    """
    words_str = ""
    for word in utterance:
        if len(word) <= 1:
            words_str += word
        else:
            words_str += " " + word
    
    # cleaned_text = regex_pipeline(words_str)
    return words_str.strip()

def replace_and_modify(row: pd.Series, pattern: str, replacement: str, cleaning_operation: str) -> pd.Series:

    # perform main cleaning operation
    row['utterance'] = re.sub(pattern, replacement, row['utterance'], flags=re.IGNORECASE)

    # track the modification
    row['utterance_modifications'].append(cleaning_operation)

    # return the modified row
    return row

UNDERSCORE_REGEX_PATTERN = re.compile(r'\S*_\S*')

def code_switching_cleaning(row: pd.Series) -> pd.Series:
    code_switches = re.findall(UNDERSCORE_REGEX_PATTERN, row['utterance'])

    for code_switch in code_switches:
        # Replace underscores with spaces
        cleaned_phrase = code_switch.replace('_', ' ')
        
        # Update the utterance
        row['utterance'] = row['utterance'].replace(code_switch, cleaned_phrase)
        
        # Track the modification
        row['utterance_modifications'].append(f"CODE SWITCHING: '{code_switch}' -> '{cleaned_phrase}'")

    return row

def update_master_df(master_df: pd.DataFrame, modified_rows: pd.DataFrame) -> pd.DataFrame:
    # set_index without inplace returns a new DataFrame with the updated index
    master_idx = master_df.set_index('utterance_id')
    modified_idx = modified_rows.set_index('utterance_id')

    # .update() operates in-place on the caller (master_idx)
    master_idx.update(modified_idx)

    # Reset the index to restore 'utterance_num' as a regular column
    return master_idx.reset_index()

def normalize_whitespace(df: pd.DataFrame, col: str = 'utterance',
                         note: str = "WHITESPACE NORMALIZED") -> pd.DataFrame:
    original = df[col]
    cleaned  = original.str.replace(r'\s+', ' ', regex=True).str.strip()
    changed  = original.notna() & cleaned.ne(original)

    df.loc[changed, col] = cleaned[changed]
    df.loc[changed, 'utterance_modifications'] = (
        df.loc[changed, 'utterance_modifications'].map(lambda mods: mods + [note])
    )
    return df

def build_document_and_audit(subset: pd.Series) -> pd.Series:
    # 1. Sort the subset descending by utterance_num
    sorted_subset = subset.sort_values('utterance_num', ascending=True)
    
    # 2. Filter out NaN and empty strings
    valid_mask = (
        sorted_subset['utterance'].notna() & 
        (sorted_subset['utterance'].astype(str).str.strip() != "")
    )
    # Use .copy() to prevent SettingWithCopyWarning when we add the new column
    filtered = sorted_subset[valid_mask].copy() 
    
    # 3. RECALCULATE: Create a contiguous line number for the reconstructed doc
    filtered['doc_line_num'] = range(1, len(filtered) + 1)
    
    # 4. Reconstruct Document
    formatted_lines = (
        filtered['speaker'].astype(str) + 
        ": " + 
        filtered['utterance'].astype(str)
    )
    reconstructed_doc = '\n'.join(formatted_lines)
    
    # 5. Extract parallel lists
    cleaned_list = filtered['utterance'].tolist()
    original_list = filtered['original_utterance'].tolist()
    
    # 6. Build a Two-Way Audit Trail
    # Format: "Doc Line X (Raw ID Y) - Speaker: [modifications]"
    audit_trail = [
        f"Doc Line {doc_num} (Raw ID {int(raw_id)}) - {spk}: {mods}" 
        for doc_num, raw_id, spk, mods in zip(
            filtered['doc_line_num'],
            filtered['utterance_num'], 
            filtered['speaker'], 
            filtered['utterance_modifications']
        )
        if len(mods) > 0  # Only include entries with modifications
    ]
    
    return pd.Series({
        'reconstructed_document': reconstructed_doc,
        'cleaned_utterances': cleaned_list,
        'original_utterances': original_list,
        'audit_trail': audit_trail
    })