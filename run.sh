uv sync

mkdir -p logs

# 1) questions
uv run python -u -m agents.question_agent --num_questions 40 --batch_size 5 \
  2>&1 | tee "logs/question_$(date +'%Y-%m-%d_%H-%M-%S').log"

# 2) answers
uv run python -u -m agents.answer_agent --input_file outputs/filtered_questions.json --batch_size 5 \
  2>&1 | tee "logs/answer_$(date +'%Y-%m-%d_%H-%M-%S').log"

