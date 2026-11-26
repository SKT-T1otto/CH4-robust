"""Use one current public map and endpoint snapshot for Chapter 4 executor replans."""

from chapter3_bser.online.execution_manager import ExecutionManager
from core.mapping.planning_state import extract_planning_state


class CurrentEndpointExecutionManager(ExecutionManager):
    def __init__(self, env):
        self.env = env
        self._route_state = None
        self.refresh_steps = []

    @staticmethod
    def _has_endpoints(state, target):
        executor = state.agents[state.executor_id]
        expected = {f"agent_{state.executor_id}": tuple(executor.position)}
        if executor.current_navigation_target is not None and target == tuple(executor.current_navigation_target):
            expected[f"navigation_target_{state.executor_id}"] = target
        # A registered, invalid endpoint is current too. The native route query
        # must keep its invalid/blocked result even when no connector exists.
        endpoints = {item.endpoint_id: item.point for item in state.planning_graph.endpoint_connectors
                     if item.role == "executor"}
        return all(endpoints.get(name) == point for name, point in expected.items())

    def _assignment(self, state, target_region, *, source):
        target = tuple(float(value) for value in target_region)
        if state.target_found and not self._has_endpoints(state, target):
            if self._route_state is None or self._route_state.step != state.step:
                fresh = extract_planning_state(self.env)
                if fresh.step != state.step or fresh.agents[state.executor_id] != state.agents[state.executor_id]:
                    raise RuntimeError("executor route query and live public state differ")
                self._route_state = fresh
                self.refresh_steps.append(state.step)
            state = self._route_state
        return super()._assignment(state, target, source=source)


def install_executor_navigation(runtime):
    """Opt in once per mission, before the first post-reset action."""
    allocator = runtime.controller.allocator
    if isinstance(allocator.execution, CurrentEndpointExecutionManager):
        return allocator.execution
    if runtime.step != 0 or type(allocator.execution) is not ExecutionManager:
        raise ValueError("expected a freshly initialized reference executor allocator")
    allocator.execution = CurrentEndpointExecutionManager(runtime.env)
    return allocator.execution
