PY=uv run

# File patterns for Python files to be included in format and linting
FILE_PATTERNS=./benchmarks ./datasets ./metrics ./models ./prompts ./utils

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
	@echo "Tests directory not yet implemented. Add tests to enable 'make test'."

.PHONY: install
install:   # Configure uv, precommit
	uv sync
	pre-commit install

.PHONY: typecheck
typecheck:  # Check for type errors in Python code.
	$(PY) mypy ./benchmarks ./models

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

# ============================================================================
# AWS Authentication & Credentials
# ============================================================================

.PHONY: list-aws-profiles
list-aws-profiles:  # List all configured AWS profiles
	@echo "Configured AWS profiles:"
	@grep "\[profile " ~/.aws/config | sed 's/\[profile //g' | sed 's/\]//g' || echo "No profiles found"
	@echo "\nDefault profile:"
	@grep "\[default\]" ~/.aws/config > /dev/null && echo "default" || echo "Not configured"

.PHONY: aws-whoami
aws-whoami:  # Show current AWS identity
	@echo "Current AWS identity:"
	@aws sts get-caller-identity 2>/dev/null || echo "Not authenticated. Run 'aws sso login' or configure credentials."

.PHONY: aws-whoami-profile
aws-whoami-profile:  # Show AWS identity for specific profile (usage: make aws-whoami-profile PROFILE=eu-dev)
	@if [ -z "$(PROFILE)" ]; then \
		echo "Usage: make aws-whoami-profile PROFILE=<profile-name>"; \
		exit 1; \
	fi
	@echo "AWS identity for profile $(PROFILE):"
	@aws sts get-caller-identity --profile $(PROFILE) 2>/dev/null || echo "Not authenticated for profile $(PROFILE)"

.PHONY: bedrock-test
bedrock-test:  # Test Bedrock connection with Claude (usage: make bedrock-test PROFILE=default)
	@PROFILE=$${PROFILE:-default}; \
	echo "Testing Bedrock with profile: $$PROFILE"; \
	$(PY) python sandbox/llm_call.py --client bedrock \
		--model anthropic.claude-3-5-haiku-20241022-v1:0 \
		--prompt "Say 'Hello from AWS Bedrock!' in one sentence." \
		--profile $$PROFILE

.PHONY: refresh-aws-credentials
refresh-aws-credentials:  # Refresh AWS credentials from Secrets Manager (requires AWS authentication)
	@echo "Fetching AWS credentials from Secrets Manager..."
	@if [ -z "$(SECRET_ID)" ]; then \
		echo "Usage: make refresh-aws-credentials SECRET_ID=<secret-id> [PROFILE=<profile>]"; \
		echo "Example: make refresh-aws-credentials SECRET_ID=research/aws-creds PROFILE=eu-dev"; \
		exit 1; \
	fi
	@PROFILE_FLAG=""; \
	if [ -n "$(PROFILE)" ]; then \
		PROFILE_FLAG="--profile $(PROFILE)"; \
	fi; \
	aws secretsmanager $$PROFILE_FLAG get-secret-value \
		--secret-id $(SECRET_ID) \
		--query 'SecretString' \
		--output text | jq -r 'to_entries | .[] | "\(.key)=\(.value)"' >> .env.local
	@echo "Credentials appended to .env.local"

.PHONY: setup-env
setup-env:  # Create .env.local from template
	@if [ ! -f .env.local ]; then \
		cp .env.example .env.local; \
		echo "Created .env.local from template. Edit it with your credentials."; \
	else \
		echo ".env.local already exists. To recreate, delete it first."; \
	fi

.PHONY: setup-env-aws
setup-env-aws:  # [OPTIONAL] Setup environment variables from AWS Secrets Manager (requires AWS profile)
	@PROFILE=$${PROFILE:-eu-dev}; \
	echo "Setting up .env.local with Azure credentials using profile: $$PROFILE..."; \
	{ \
	  azure_ad_secret=$$(aws secretsmanager --profile $$PROFILE get-secret-value --secret-id chat/azure-ad-local --query 'SecretString' | jq -r 'fromjson'); \
	  azure_sp_secret=$$(aws secretsmanager --profile $$PROFILE get-secret-value --secret-id chat/azure-service-principal --query 'SecretString' | jq -r 'fromjson'); \
	  azure_sandbox_secret=$$(aws secretsmanager --profile $$PROFILE get-secret-value --secret-id azure-sandbox-creds --query 'SecretString' | jq -r 'fromjson'); \
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