use agent::AgentConfig;
use multi_agent::MultiAgentSystem;
use rand::Rng;

fn main() {
    let config = AgentConfig {
        input_size: 10,
        hidden_size: 50,
        output_size: 1,
    };

    let system = MultiAgentSystem::new(10, &config);

    // Generate dummy data
    let mut rng = rand::thread_rng();
    let inputs: Vec<Vec<f64>> = (0..10).map(|_| (0..10).map(|_| rng.gen::<f64>()).collect()).collect();
    let targets: Vec<Vec<f64>> = (0..10).map(|_| vec![rng.gen::<f64>()]).collect();

    // Training
    for _ in 0..100 {
        system.parallel_train(inputs.clone(), targets.clone(), config.input_size, config.output_size);
    }

    // Prediction
    let predictions = system.predict(inputs, config.input_size);
    for prediction in predictions {
        println!("{:?}", prediction);
    }
}
