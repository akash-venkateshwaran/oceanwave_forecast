import mlflow
from mlflow.entities import Experiment, ViewType
from mlflow.tracking import MlflowClient
from loguru import logger
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import torch

from .config import PROJ_ROOT


class MLflowExperimentManager:
    """
    Manages MLflow experiments and runs, providing utilities for creation, logging, and retrieval.
    
    """

    def __init__(self, experiment_name: str, run_number: int, tags: Optional[Dict[str, str]] = None):
        """
        Initializes the MLflowExperimentManager.

        Args:
            experiment_name: The name of the MLflow experiment.
            run_number: A unique identifier for the run within the experiment.
            tags: Optional dictionary of tags to associate with the experiment.
        """
        if not experiment_name:
            raise ValueError("Experiment name cannot be empty.")
        if not isinstance(run_number, int) or run_number < 0:
            raise ValueError("Run number must be non-negative integer.")

        # Set tracking URI first
        self._set_tracking_uri()
        
        self.experiment_name = experiment_name
        self.run_number = run_number
        self.tags = tags if tags is not None else {}
        self.client = MlflowClient()
        self._current_run = None

    def _set_tracking_uri(self):
            """
            Alternative solution using file:// with proper URL encoding.
            """
            from urllib.parse import quote
            
            mlruns_path = PROJ_ROOT / 'mlruns'
            mlruns_path.mkdir(parents=True, exist_ok=True)
            
            # URL encode the path to handle spaces
            encoded_path = quote(str(mlruns_path.absolute()), safe=':/')
            tracking_uri = f"file:///{encoded_path}"
            
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")



    @staticmethod
    def get_or_create_experiment(name: str, tags: Dict[str, str]) -> Experiment:
        """
        Returns an experiment whose *name* == `name` AND whose tag set ⊇ `tags`.
        If none exists (or tags differ), creates a NEW experiment with a unique
        name derived from `name`.

        Args:
            name: The desired name for the experiment.
            tags: A dictionary of tags to match or apply to the experiment.

        Returns:
            An MLflow Experiment object.
        """
        exp_list = mlflow.search_experiments(
            filter_string=f"name LIKE '{name}%'",
            view_type=ViewType.ALL,
        )

        # Try exact match on name AND tags
        for exp in exp_list:
            if exp.name == name:
                logger.info(f"Found existing experiment '{name}' with ID {exp.experiment_id}.")
                return exp

        # Otherwise, invent a unique name
        suffix = 1
        candidate_name = name
        existing_names = {e.name for e in exp_list}
        while candidate_name in existing_names:
            suffix += 1
            candidate_name = f"{name}_{suffix}"

        try:
            exp_id = mlflow.create_experiment(candidate_name, tags=tags)
            logger.info(f"Created new experiment '{candidate_name}' with ID {exp_id}.")
            return mlflow.get_experiment(exp_id)
        except Exception as e:
            logger.error(f"Failed to create experiment '{candidate_name}': {e}")
            raise

    def start_mlflow_run(self, run_name_prefix: str = "run") -> mlflow.ActiveRun:
        """
        Starts an MLflow run, handling existing runs and resumption.

        Args:
            run_name_prefix: Prefix for the run name (e.g., "run").

        Returns:
            An active MLflow run object.
        """
        experiment = self.get_or_create_experiment(self.experiment_name, self.tags)

        existing_runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"params.run_number = '{self.run_number}'",
            run_view_type=ViewType.ALL,
            max_results=1,
        )

        resume_id = None
        if existing_runs and existing_runs[0].info.status == "FINISHED":
            self._current_run = existing_runs[0]
            logger.info(f"Finished run already logged (id={self._current_run.info.run_id}) – skipping.")
            return self._current_run
        elif existing_runs:
            resume_id = existing_runs[0].info.run_id
            logger.info(f"Resuming existing run (id={resume_id}).")

        try:
            self._current_run = mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"{run_name_prefix}_{self.run_number:03d}",
                run_id=resume_id,
            )
            logger.info(f"Started MLflow run with ID: {self._current_run.info.run_id}")
            # Log run_number as a parameter if it's a new run or being resumed without it
            if not self._current_run.data.params.get("run_number"):
                mlflow.log_param("run_number", self.run_number)
            return self._current_run
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise

    def end_mlflow_run(self):
        """Ends the current MLflow run."""
        if self._current_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run with ID: {self._current_run.info.run_id}")
            self._current_run = None
        else:
            logger.warning("No active MLflow run to end.")

    def log_params(self, params: Dict[str, any]):
        """Logs a dictionary of parameters to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log parameters.")
            return
        try:
            mlflow.log_params(params)
            logger.info(f"Logged parameters: {params.keys()}")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_param(self, key: str, value: any):
        """Logs a single parameter to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log parameter.")
            return
        try:
            mlflow.log_param(key, value)
            logger.info(f"Logged parameter: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")

    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Logs a single metric to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log metric.")
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metric {key}: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs a dictionary of metrics to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log metrics.")
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs a local file or directory as an artifact to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log artifact.")
            return
        if not Path(local_path).exists():
            logger.error(f"Artifact path does not exist: {local_path}")
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path} to {artifact_path or 'root'}")
        except Exception as e:
            logger.error(f"Failed to log artifact {local_path}: {e}")

    def log_pytorch_model(self, pytorch_model: torch.nn.Module, artifact_path: str = "model"):
        """Logs a PyTorch model to the current MLflow run."""
        if not self._current_run:
            logger.warning("No active MLflow run. Cannot log PyTorch model.")
            return
        try:
            import torch # Import torch here to avoid circular dependency if not used elsewhere
            mlflow.pytorch.log_model(pytorch_model, artifact_path=artifact_path)
            logger.info(f"Logged PyTorch model to artifact path: {artifact_path}")
        except ImportError:
            logger.error("PyTorch is not installed. Cannot log PyTorch model.")
        except Exception as e:
            logger.error(f"Failed to log PyTorch model: {e}")

    @staticmethod
    def get_run_details(experiment_name: str, run_number: int) -> Optional[mlflow.entities.Run]:
        """
        Retrieves details of a specific run within an experiment.

        Args:
            experiment_name: The name of the MLflow experiment.
            run_number: The run number to search for.

        Returns:
            An MLflow Run object if found, otherwise None.
        """
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{experiment_name}' not found.")
            return None

        runs = client.search_runs(
            experiment.experiment_id,
            filter_string=f"params.run_number = '{run_number}'",
            order_by=["attribute.start_time DESC"],
            max_results=1
        )

        if not runs:
            logger.info(f"No run found with run_number '{run_number}' in experiment '{experiment_name}'.")
            return None
        
        run = runs[0]
        logger.info(f"Found run {run.info.run_id} for experiment '{experiment_name}' and run_number {run_number}.")
        return run

    @staticmethod
    def get_metric_history(run_id: str, metric_key: str) -> List[mlflow.entities.Metric]:
        """
        Retrieves the history of a specific metric for a given run.

        Args:
            run_id: The ID of the MLflow run.
            metric_key: The name of the metric.

        Returns:
            A list of MLflow Metric objects, sorted by step.
        """
        client = MlflowClient()
        try:
            history = client.get_metric_history(run_id, metric_key)
            history.sort(key=lambda m: m.step)
            logger.info(f"Retrieved history for metric '{metric_key}' from run {run_id}.")
            return history
        except Exception as e:
            logger.error(f"Failed to retrieve metric history for run {run_id}, metric {metric_key}: {e}")
            return []

    @staticmethod
    def search_experiments_by_tag(tag_key: str, tag_value: str) -> List[Experiment]:
        """
        Searches for MLflow experiments by a specific tag key and value.

        Args:
            tag_key: The key of the tag.
            tag_value: The value of the tag.

        Returns:
            A list of MLflow Experiment objects matching the criteria.
        """
        try:
            experiments = mlflow.search_experiments(
                view_type=ViewType.ALL,
                filter_string=f"tags.{tag_key} = '{tag_value}'",
            )
            logger.info(f"Found {len(experiments)} experiments with tag '{tag_key}'='{tag_value}'.")
            return experiments
        except Exception as e:
            logger.error(f"Failed to search experiments by tag {tag_key}={tag_value}: {e}")
            return []

    @staticmethod
    def list_runs_in_experiment(experiment_id: str) -> List[mlflow.entities.Run]:
        """
        Lists all runs within a given MLflow experiment ID.

        Args:
            experiment_id: The ID of the experiment.

        Returns:
            A list of MLflow Run objects.
        """
        client = MlflowClient()
        try:
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ALL
            )
            logger.info(f"Found {len(runs)} runs in experiment {experiment_id}.")
            return runs
        except Exception as e:
            logger.error(f"Failed to list runs for experiment {experiment_id}: {e}")
            return []

    @staticmethod
    def _print_run_tree(run: mlflow.entities.Run, children_map: Dict[str, List[mlflow.entities.Run]], level: int = 0):
        """Helper to pretty-print run hierarchy."""
        pad = "  " * level
        name = run.data.tags.get("mlflow.runName", "")
        logger.info(f"{pad}• {name or run.info.run_id} (run_id={run.info.run_id}, status={run.info.status})")
        for child in children_map.get(run.info.run_id, []):
            MLflowExperimentManager._print_run_tree(child, children_map, level + 1)

    @staticmethod
    def print_experiment_runs_tree(experiment_id: str):
        """
        Prints the run hierarchy for a given experiment ID.

        Args:
            experiment_id: The ID of the experiment.
        """
        runs = MLflowExperimentManager.list_runs_in_experiment(experiment_id)

        # Build parent → children map
        children_map, roots = {}, []
        for r in runs:
            parent = r.data.tags.get("mlflow.parentRunId")
            (children_map.setdefault(parent, []).append(r)
             if parent else roots.append(r))

        # Pretty-print
        logger.info(f"\n=== Runs for Experiment ID: {experiment_id} ===")
        for r in roots:
            MLflowExperimentManager._print_run_tree(r, children_map)