#!/bin/bash

# Safely replace one dotenv key without treating the value as a sed expression.
set_env_value() {
    local env_file="$1"
    local env_key="$2"
    local env_value="$3"
    local env_tmp
    local env_found="false"
    local env_line

    if [[ "$env_value" == *$'\n'* || "$env_value" == *$'\r'* ]]; then
        echo "エラー: ${env_key} に改行は使用できません" >&2
        return 1
    fi

    env_tmp=$(mktemp "${env_file}.XXXXXX")
    chmod 0600 "$env_tmp"
    while IFS= read -r env_line || [ -n "$env_line" ]; do
        if [[ "$env_line" == "${env_key}="* ]]; then
            printf '%s=%s\n' "$env_key" "$env_value" >> "$env_tmp"
            env_found="true"
        else
            printf '%s\n' "$env_line" >> "$env_tmp"
        fi
    done < "$env_file"
    if [ "$env_found" != "true" ]; then
        printf '%s=%s\n' "$env_key" "$env_value" >> "$env_tmp"
    fi
    mv "$env_tmp" "$env_file"
    chmod 0600 "$env_file"
}
