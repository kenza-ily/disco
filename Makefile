PY=uv run

# File patterns for Python files to be included in format and linting
FILE_PATTERNS=./app ./tests

# Add a configurable break between commands
BR=&& echo "\n---\n"

# Help is based on: https://news.ycombinator.com/item?id=11195539
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: all
all:  # Prepare code for committing to git. Note: suppresses "typecheck" failures
	$(MAKE) format $(BR)
	$(MAKE) lint $(BR)
	$(MAKE) test $(BR)
	$(MAKE) typecheck $(BR)

.PHONY: test
test:  # Test Python code. See additional settings in pyproject.toml
	$(PY) pytest ./tests

.PHONY: install
install:   # Configure uv, precommit
	uv sync
	pre-commit install

.PHONY: typecheck
typecheck:  # Check for type errors in Python code.
	$(PY) mypy ./app

.PHONY: lint
lint:  ## Run linters
	$(PY) ruff check $(FILE_PATTERNS) $(BR)

.PHONY: format
format:  # Automatically format python code. WARNING: unsaved changes may be lost
	$(PY) ruff format $(FILE_PATTERNS) $(BR)
	$(PY) ruff check --fix $(FILE_PATTERNS) $(BR)

.PHONY: update
update:  # Chore for minor updates to be used alongside manual changes to pyproject.toml
	pre-commit autoupdate $(BR)
	uv sync $(BR)

.PHONY: cov
cov: ## Run tests with coverage
	uv run pytest --cov=app --cov-report=term-missing

.PHONY: setup-env
setup-env:  # Setup environment variables from AWS secrets
	@echo "Setting up .env.local with Azure credentials..."
	@{ \
	  azure_ad_secret=$$(aws secretsmanager --profile eu-dev get-secret-value --secret-id chat/azure-ad-local --query 'SecretString' | jq -r 'fromjson'); \
	  azure_sp_secret=$$(aws secretsmanager --profile eu-dev get-secret-value --secret-id chat/azure-service-principal --query 'SecretString' | jq -r 'fromjson'); \
	  azure_sandbox_secret=$$(aws secretsmanager --profile eu-dev get-secret-value --secret-id azure-sandbox-creds --query 'SecretString' | jq -r 'fromjson'); \
	  echo "\
	AZURE_AD_CLIENT_ID=$$(echo $$azure_ad_secret | jq -r '.AZURE_AD_CLIENT_ID')\n\
	AZURE_AD_CLIENT_SECRET=$$(echo $$azure_ad_secret | jq -r '.AZURE_AD_CLIENT_SECRET')\n\
	AZURE_AD_TENANT_ID=$$(echo $$azure_ad_secret | jq -r '.AZURE_AD_TENANT_ID')\n\
	AZURE_CLIENT_ID=$$(echo $$azure_sp_secret | jq -r '.AZURE_CLIENT_ID')\n\
	AZURE_CLIENT_SECRET=$$(echo $$azure_sp_secret | jq -r '.AZURE_CLIENT_SECRET')\n\
	AZURE_TENANT_ID=$$(echo $$azure_sp_secret | jq -r '.AZURE_TENANT_ID')\n\
	AZURE_OPENAI_API_KEY=$$(echo $$azure_sandbox_secret | jq -r '.AZUREAI_API_KEY')\n\
	AZURE_OPENAI_ENDPOINT=https://ai-pxlailabsbxhub741529278955.openai.azure.com/\n\
	AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://ai-pxlailabsbxhub741529278955.cognitiveservices.azure.com/\n\
	AZURE_DOCUMENT_INTELLIGENCE_KEY=$$(echo $$azure_sandbox_secret | jq -r '.AZUREAI_API_KEY')" > .env.local; \
	}
	@echo ".env.local created successfully."