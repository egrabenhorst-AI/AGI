use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device};
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize, Debug)]
pub struct AgentConfig {
    pub input_size: i64,
    pub hidden_size: i64,
    pub output_size: i64,
}

pub struct Agent {
    model: nn::Sequential,
    optimizer: Arc<Mutex<nn::Optimizer<nn::Adam>>>,
    device: Device,
}

unsafe impl Send for Agent {}
unsafe impl Sync for Agent {}

impl Agent {
    pub fn new(config: &AgentConfig) -> Self {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let model = nn::seq()
            .add(nn::linear(&vs.root(), config.input_size, config.hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root(), config.hidden_size, config.output_size, Default::default()));

        let optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();

        Agent {
            model,
            optimizer: Arc::new(Mutex::new(optimizer)),
            device,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        self.model.forward(input)
    }

    pub fn train(&self, input: &Tensor, target: &Tensor) {
        let output = self.forward(input);
        let loss = output.mse_loss(target, tch::Reduction::Mean);
        let mut optimizer = self.optimizer.lock().unwrap();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
