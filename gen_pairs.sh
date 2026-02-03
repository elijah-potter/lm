models=(
"ministral-3:3b"
"gemma3n:e4b"
)

pronouns=(
"I"
"me"
"my"
"mine"
"we"
"us"
"our"
"ours"
"you"
"your"
"yours"
"he"
"him"
"his"
"she"
"her"
"hers"
"it"
"its"
"they"
"them"
"their"
)

while [[ true ]];
do
  for model in "${models[@]}";
  do
    for pronoun in "${pronouns[@]}";
    do
      ofc -m "$model" -t 0.8 "Write a sentence in the passive voice, then follow it up with the same sentence, written in the active voice. Do not use any formatting. Do not say anything beside the requested sentence. Use the word $pronoun somewhere." | tee ./test_pairs/`uuidgen`.md
  done
  done
done
