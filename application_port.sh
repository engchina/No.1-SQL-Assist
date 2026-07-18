#!/bin/bash

DEFAULT_APPLICATION_PORT=8080
APPLICATION_PORT_FILE="${APPLICATION_PORT_FILE:-/u01/aipoc/props/application_port.txt}"

resolve_application_port() {
    local application_port

    if [ -n "${APPLICATION_PORT:-}" ]; then
        application_port="$APPLICATION_PORT"
    elif [ -r "$APPLICATION_PORT_FILE" ]; then
        IFS= read -r application_port < "$APPLICATION_PORT_FILE"
    else
        application_port="$DEFAULT_APPLICATION_PORT"
    fi

    case "$application_port" in
        ""|*[!0-9]*)
            echo "Invalid application port: $application_port" >&2
            return 1
            ;;
    esac

    if [ "$application_port" -lt 1 ] || [ "$application_port" -gt 65535 ]; then
        echo "Application port must be between 1 and 65535: $application_port" >&2
        return 1
    fi

    printf '%s\n' "$application_port"
}
