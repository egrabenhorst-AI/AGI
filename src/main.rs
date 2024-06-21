use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task;

const NUM_AGENTS: usize = 10;
const NUM_EPISODES: usize = 1000;
const ALPHA: f64 = 0.1;
const GAMMA: f64 = 0.9;
const EPSILON: f64 = 0.1;
const NUM_STATES: usize = 100;
const NUM_ACTIONS: usize = 4;

struct Agent {
    q_table: Vec<Vec<f64>>,
    id: usize,
}

impl Agent {
    fn new(id: usize) -> Self {
        Self {
            q_table: vec![vec![0.0; NUM_ACTIONS]; NUM_STATES],
            id,
        }
    }

    fn select_action(&self, state: usize) -> usize {
        if rand::thread_rng().gen::<f64>() < EPSILON {
            rand::thread_rng().gen_range(0..NUM_ACTIONS)
        } else {
            let actions = &self.q_table[state];
            actions.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        }
    }

    fn update_q_value(&mut self, state: usize, action: usize, reward: f64, next_state: usize) {
        let best_next_action = self.q_table[next_state].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let q_value = &mut self.q_table[state][action];
        *q_value += ALPHA * (reward + GAMMA * best_next_action - *q_value);
    }

    fn run(&mut self, tx: mpsc::Sender<(usize, usize, f64, usize)>, rx: Arc<Mutex<mpsc::Receiver<(usize, usize, f64, usize)>>>) {
        let mut rx = rx.lock().unwrap();
        let mut state = rand::thread_rng().gen_range(0..NUM_STATES);
        for _ in 0..NUM_EPISODES {
            let action = self.select_action(state);
            let next_state = (state + action) % NUM_STATES; // Example transition logic
            let reward = if next_state == NUM_STATES - 1 { 1.0 } else { 0.0 }; // Example reward logic
            self.update_q_value(state, action, reward, next_state);
            state = next_state;
            let _ = tx.blocking_send((self.id, state, reward, next_state));
            // Receive messages
            while let Ok((_, _, _, _)) = rx.try_recv() {
                // Process received messages if needed
            }
        }
    }
    
    
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(100);
    let rx = Arc::new(Mutex::new(rx));
    let agents: Vec<Arc<Mutex<Agent>>> = (0..NUM_AGENTS).map(|id| Arc::new(Mutex::new(Agent::new(id)))).collect();

    let handles: Vec<_> = agents.iter().cloned().map(|agent| {
        let tx = tx.clone();
        let rx = Arc::clone(&rx);
        task::spawn_blocking(move || agent.lock().unwrap().run(tx, rx))
    }).collect();

    for handle in handles {
        handle.await.unwrap();
    }

    // Gather results or further process the Q-tables as needed
}

