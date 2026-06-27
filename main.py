import helper_functions as helper

import pylangacq
import pandas as pd

chats = pylangacq.read_chat("spa.zip")

master_df = pd.DataFrame(columns=["file_path", "utterance_num", "utterance", "speaker"])

for chat_file_path, file_utterances in zip(chats.file_paths, chats.utterances(by_file=True)):
    
    raw_words = []
    utterance_words = []
    speaker_map = []

    temp_df = pd.DataFrame(columns=["file_path", "utterance_num", "utterance", "speaker"])
    
    # Iterate through the actual Utterance objects
    for u in file_utterances:
        # 2. Extract the clean words from the utterance's tokens
        words = [token.word for token in u.tokens]        

        words_str = helper.clean_utterance(words)
        
        # check if this is an orphaned grammatical marker (e.g., a single punctuation mark) and skip it if so
        if len(words_str.strip()) <= 1:
            continue

        cleaned_utterance = helper.clean_utterance(words_str.strip())

        utterance_words.append(cleaned_utterance)
        # Get the speaker for this specific utterance (e.g., 'CHI', 'MOT')
        speaker_map.append(u.participant)

    temp_df = pd.DataFrame({
        "file_path": [chat_file_path] * len(utterance_words),
        "utterance_num": list(range(len(utterance_words))),
        "utterance": utterance_words,
        "speaker": speaker_map
    })

    master_df = pd.concat([master_df, temp_df], ignore_index=True)

# create unique utterance IDs
master_df.reset_index(names='utterance_id', inplace=True)

# words to drop:
# words are surrounded by a regex pattern to ensure we do not create spacing issues (ie leaving double spaces)
# leave "background noise" as it needs special treatment
annotation_drop_filter = {
    "crying": r'\bcrying\b',
    "clear": r'\bclear\b',
    "vocalization": r'\bvocalization\b',
    "noise": r'\bnoise\b',
}

symbol_drop_filter = {
    "+...": (r'\s*\+\.\.\.', "..."),
    "+": (r'\+', "")
}

# track original utterances
original_utterances = master_df['utterance'].copy()
master_df['original_utterance'] = original_utterances

# track utterance modifications
master_df['utterance_modifications'] = [[] for _ in range(len(master_df))]

# first handle code switching
code_switching_rows = master_df[master_df['utterance'].str.contains('_', na=False)]

code_switching_rows = code_switching_rows.apply(helper.code_switching_cleaning, axis=1)

# print(type(code_switching_rows))

# update master df
master_df = helper.update_master_df(master_df, code_switching_rows)

# second filter and clean "background noise" rows
background_noise_rows = master_df[master_df['utterance'].str.contains("background noise", case=False, na=False)]

background_noise_rows = background_noise_rows.apply(lambda row: helper.replace_and_modify(row, r"background noise", "", "ANNOTATION ARTIFACT REMOVED: 'background noise'"), axis=1)

# update master df
master_df = helper.update_master_df(master_df, background_noise_rows)

# third filter out all other annotation artifacts rows only if artifact is surrounded by whitespace or at the start/end of the utterance (preserves these strings showing up in long Spanish words)

for artifact, pattern in annotation_drop_filter.items():
    artifact_rows = master_df[master_df['utterance'].str.contains(pattern, na=False)]
    
    artifact_rows = artifact_rows.apply(lambda row: helper.replace_and_modify(row, pattern, " ", f"ANNOTATION ARTIFACT REMOVED: ' {artifact} '"), axis=1)
    
    # update master df
    master_df = helper.update_master_df(master_df, artifact_rows)

# now take care of `+/.` artifacts first then run another clean for stray `+` that can be a part of `+...`
for artifact, (pattern, replacement) in symbol_drop_filter.items():
    artifact_rows = master_df[master_df['utterance'].str.contains(pattern, na=False)]
    
    artifact_rows = artifact_rows.apply(lambda row, p=pattern, r=replacement: 
                                        helper.replace_and_modify(row, p, r, f"ANNOTATION SYMBOL NORMALIZED/REMOVED: ' {artifact} '"), 
                                        axis=1
                                    )
    
    # update master df
    master_df = helper.update_master_df(master_df, artifact_rows)

# pass to clean up whitepace
# collapse spaces on all utterances
master_df = helper.normalize_whitespace(master_df, col='utterance', note="WHITESPACE NORMALIZATION")

# create final dataframe with document and audit trail
final_df = (
    master_df.groupby('file_path')
    .apply(helper.build_document_and_audit)
    .reset_index() 
)

# export to parquet
final_df.to_parquet("callhome_es_cleaned_transcript_pretraining_docs.parquet")