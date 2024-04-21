check_dirs := visprompt tests

# This target runs checks on all files and potentially modifies some of them
style:
	poetry run isort $(check_dirs)
	poetry run black $(check_dirs)
	poetry run flake8 $(check_dirs)

# Run tests for the library
test:
	poetry run coverage erase
	poetry run coverage run -a --source=./visprompt --branch -m pytest -s -v --black --isort tests --junit-xml unit_results.xml
	poetry run coverage report
	poetry run coverage xml
