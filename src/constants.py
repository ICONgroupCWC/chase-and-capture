SpiderAndFlyEnv = 'PredatorPrey10x10-v4'
BaselineModelPath_10x10_4v2 = 'artifacts/baseline_policy_10x10_4v2.pt'
BaselineModelPath_10x10_4v3 = 'artifacts/baseline_policy_10x10_4v3.pt'
RolloutModelPath_10x10_4v2 = 'artifacts/rollout_policy_10x10_4v2.pt'
RepeatedRolloutModelPath_10x10_4v2 = 'artifacts/repeated_rollout_policy_10x10_4v2.pt'
RepeatedRolloutModelPath_10x10_4v3 = 'artifacts/repeated_rollout_policy_10x10_4v3.pt'
RepeatedRolloutModelPath_10x10_4v4 = 'artifacts/repeated_rollout_policy_10x10_4v4.pt'
deter_5050Prey = 'artifacts/deter_5050_mode11.pt'
deter_5050Prey_Cross_13 = 'artifacts/deter_5050_mode13.pt'
deter_5050Prey_Cross_12 = 'artifacts/deter_5050_mode12.pt'
random_5050Prey = 'artifacts/random_5050.pt'
evolvedWM = 'artifacts/evolved_11_on_15_ITER_2.pt'
deter_5050Prey_agent_0 = 'artifacts/deter_5050_mode15_agent_0.pt'
deter_5050Prey_agent_1 = 'artifacts/deter_5050_mode15_agent_1.pt'
deter_11_on_20 = 'artifacts/deter_5050_mode11_agent_1.pt  TRAINED'


l1_loss_agent_1 = 'artifacts/L1_mode20_agent_1.pt'
ce_loss_agent_1 = 'artifacts/CE_mode20_agent_1.pt'
mse_loss_agent_1 = 'artifacts/MSE_mode20_agent_1.pt'



class AgentType:
    RANDOM = 'Random'
    RULE_BASED = 'Rule-Based'  # Smallest Manhattan Distance
    QNET_BASED = 'QNet-Based'
    SEQ_MA_ROLLOUT = 'Agent-by-agent MA Rollout'
    STD_MA_ROLLOUT = 'Standard MA Rollout'


class QnetType:
    BASELINE = 'Trained from Rollout Data'
    REPEATED = 'Trained from Repeated Rollout Data'
