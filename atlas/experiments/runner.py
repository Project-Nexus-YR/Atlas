import time
from typing import Any, Dict, Optional

from atlas.agents.base import Agent
from atlas.configs.settings import settings
from atlas.experiments.logging import ExperimentLogger


class ExperimentRunner:
    """
    Orchestrates the execution of an experiment by pairing an Agent
    with a Benchmark environment, and handling the logging of results.
    """

    def __init__(self, agent: Agent, benchmark_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.agent = agent
        self.benchmark_name = benchmark_name
        self.config = config or {}
        self.logger = ExperimentLogger(project=settings.wandb_project, config=self.config)

    def run(self, max_steps: int = 10) -> Dict[str, Any]:
        """
        Executes the agent loop against the benchmark.
        """
        self.logger.start()
        
        # In a real implementation, this would load the benchmark environment
        objective = f"Solve issue from benchmark {self.benchmark_name}"
        
        start_time = time.time()
        success = False
        
        for step in range(max_steps):
            # 1. Plan
            plan = self.agent.plan(objective)
            
            # 2. Act
            action_result = self.agent.act(plan)
            
            # 3. Reflect & Learn
            reflection = self.agent.reflect(action_result)
            self.agent.learn(reflection)
            
            # Simulated environment check (dummy)
            if action_result == "success":
                success = True
                break

        end_time = time.time()
        
        # Evaluate final state
        agent_metrics = self.agent.evaluate()
        memory_stats = self.agent.memory.statistics()
        
        results = {
            "success": success,
            "duration_seconds": end_time - start_time,
            "steps_taken": step + 1,
            "agent_metrics": agent_metrics,
            "memory_stats": memory_stats
        }
        
        self.logger.log_metrics(results)
        self.logger.finish()
        
        return results
