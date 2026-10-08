#!/bin/sh
set -eu

chown -R appuser:appuser "${HF_HOME}"
exec gosu appuser "$@"
