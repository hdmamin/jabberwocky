# Run from ~/jabberwocky.
fname="NEW_PROMPT"
read -p "Prompt Name (lowercase, no spaces): " fname
fpath="data/prompts/$fname.yaml"
cp data/templates/prompt.yaml $fpath
vi $fpath
echo "Prompt saved to $fpath."
