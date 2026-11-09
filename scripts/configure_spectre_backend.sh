#!/bin/bash

# Source this file, then call configure_spectre_backend.  The HTTP mode does
# not inspect or print the bearer token, which keeps it out of xtrace logs.
configure_spectre_backend() {
    local backend="${CIRCUITOP_SPECTRE_BACKEND:-local}"
    local modulefiles_root="${1:-}"
    local module_name="${2:-SPECTRE}"

    case "${backend}" in
        local)
            if [ -n "${modulefiles_root}" ]; then
                module use "${modulefiles_root}" || {
                    echo "Failed to add Spectre modulefiles root ${modulefiles_root}"
                    return 1
                }
            fi
            module load "${module_name}" || {
                echo "Failed to load ${module_name}"
                return 1
            }
            ;;
        http)
            if [ -z "${CIRCUITOP_SPECTRE_API_URL:-}" ]; then
                echo "CIRCUITOP_SPECTRE_API_URL is required for HTTP Spectre"
                return 1
            fi
            echo "[Init] Using remote Spectre API at ${CIRCUITOP_SPECTRE_API_URL}"
            ;;
        *)
            echo "Unsupported CIRCUITOP_SPECTRE_BACKEND=${backend}; expected local or http"
            return 1
            ;;
    esac

    export CIRCUITOP_SPECTRE_BACKEND="${backend}"
}
