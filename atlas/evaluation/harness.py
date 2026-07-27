import time
from typing import Any, Dict, List
from atlas.agents.swe_agent import SWEAgent
from atlas.memory.evolution import KnowledgeEvolutionEngine
from atlas.memory.graph import GraphEcosystem
from atlas.experiments.logging import ExperimentLogger

class SWEBenchHarness:
    """
    Experimental Harness for evaluating Atlas on SWE-bench.
    Supports running baseline Vector-RAG versus the Knowledge Evolution Engine.
    """

    def __init__(self, logger: ExperimentLogger, use_kee: bool = True):
        self.logger = logger
        self.use_kee = use_kee
        # In real life, setup Docker sandbox here.

    def load_dataset(self) -> List[Dict[str, Any]]:
        # Stub: Load SWE-bench dataset (e.g. from huggingface)
        return [
            {"instance_id": "django__django-1111", "problem_statement": "Fix the parser bug..."},
            {"instance_id": "psf__requests-2222", "problem_statement": "Session timeout error..."}
        ]

    def run_evaluation(self, agent: SWEAgent):
        """Runs the agent across the dataset and logs performance."""
        self.logger.start()
        dataset = self.load_dataset()

        success_count = 0
        total_tasks = len(dataset)

        for step, instance in enumerate(dataset):
            print(f"\n[Harness] Starting instance: {instance['instance_id']}")
            
            start_time = time.time()
            plan = agent.plan(instance['problem_statement'])
            action_result = agent.act(plan)
            reflection = agent.reflect(action_result)
            agent.learn(reflection)
            
            elapsed = time.time() - start_time
            
            # Stub evaluation metric
            is_resolved = action_result.get("status") == "success"
            if is_resolved:
                success_count += 1
            
            metrics = {
                "instance_id": instance['instance_id'],
                "resolved": is_resolved,
                "time_seconds": elapsed,
                "agent_state": agent.evaluate()
            }
            self.logger.log_metrics(metrics, step=step)
            
            if self.use_kee:
                # Trigger the offline consolidation (simulating "sleep" between tasks)
                print("[Harness] Triggering offline Knowledge Consolidation...")
                agent.kee.timescales.periodic_loop(agent.kee)

        print(f"\n[Harness] Evaluation Complete. Resolves: {success_count}/{total_tasks}")
        self.logger.finish()
