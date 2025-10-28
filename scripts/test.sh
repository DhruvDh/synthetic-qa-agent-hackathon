#!/bin/bash
set -euo pipefail

python -m agents.question_agent \
    --output_file "outputs/questions.json" \
    --num_questions 20 \
    --verbose

python -m agents.answer_agent \
    --input_file "outputs/filtered_questions.json" \
    --output_file "outputs/answers.json" \
    --verbose
