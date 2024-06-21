use rayon::prelude::*;
use tch::Tensor;
use agent::{Agent, AgentConfig};
use std::sync::Arc;

pub struct MultiAgentSystem {
    agents: Vec<Arc<Agent>>,
}

impl MultiAgentSystem {
    pub fn new(num_agents: usize, config: &AgentConfig) -> Self {
        let agents = (0..num_agents).map(|_| Arc::new(Agent::new(config))).collect();
        MultiAgentSystem { agents }
    }

    pub fn parallel_train(&self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, input_size: i64, output_size: i64) {
        self.agents.par_iter().enumerate().for_each(|(i, agent)| {
            let input = Tensor::of_slice(&inputs[i]).view([input_size]);
            let target = Tensor::of_slice(&targets[i]).view([output_size]);
            agent.train(&input, &target);
        });
    }

    pub fn predict(&self, inputs: Vec<Vec<f64>>, input_size: i64) -> Vec<Vec<f64>> {
        self.agents.par_iter().zip(inputs.par_iter()).map(|(agent, input)| {
            let input_tensor = Tensor::of_slice(input).view([input_size]);
            let output_tensor = agent.forward(&input_tensor);
            let output_vec: Vec<f64> = output_tensor.into();
            output_vec
        }).collect()
    }
}
