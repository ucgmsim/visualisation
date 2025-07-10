pipeline {
    agent {
        docker {
            image 'python:3.13'
            // The -u 0 flags means run commands inside the container
            // as the user with uid = 0. This user is, by default, the
            // root user. So it is effectively saying run the commands
            // as root.
            args '-u 0'
        }
    }
    stages {
        stage('Installing OS Dependencies') {
            steps {
                echo "[[ Install GMT ]]"
                sh """
                   apt-get update
                   apt-get install -y gmt libgmt-dev libgmt6 ghostscript
                """
                echo "[[ Install Rust ]]"
                sh """
                   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                """
                echo "[[ Install uv ]]"
                sh """
                   curl -LsSf https://astral.sh/uv/install.sh | sh
                """
            }
        }
        stage('Setting up env') {
            steps {
                echo "[[ Start virtual environment ]]"
                withEnv(["PATH+RUST=$HOME/.cargo/bin", "PATH+UV=$HOME/.local/bin"]) {
                    sh """
                    cd ${env.WORKSPACE}
                    uv venv
                    uv sync
                    """
                }

            }
        }

        stage('Run regression tests') {
            steps {
                sh """
                    cd ${env.WORKSPACE}
                    source .venv/bin/activate
                    pytest --cov=visualisation --cov-report=html tests
                    python -m coverage html --skip-covered --skip-empty

                    python -m coverage report | sed 's/^/    /'
                    python -Im coverage report --fail-under=95
                """
            }
        }
    }
}
