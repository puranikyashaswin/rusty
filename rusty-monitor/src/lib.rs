use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};
use std::io::{self};
use std::time::Duration;
use sysinfo::System;

// --- App State ---

#[derive(PartialEq, Clone, Copy)]
pub enum CurrentTab {
    Dashboard,
    Chat,
    Training,
}

#[derive(PartialEq)]
pub enum InputMode {
    Normal,
    Editing,
}

#[derive(Clone)]
pub struct TrainingConfig {
    pub model_name: String,
    pub epochs: String,
    pub batch_size: String,
    pub learning_rate: String,
}

pub struct App {
    pub running: bool,
    pub active_tab: CurrentTab,
    pub sys: System,
    // Chat state
    pub chat_input: String,
    pub chat_history: Vec<(String, String)>, // (User, AI)
    // Training state
    pub training_config: TrainingConfig,
    pub active_input: usize, // 0: Model, 1: Epochs, 2: Batch, 3: LR
    pub input_mode: InputMode,
    pub is_training: bool,
    pub training_data: Vec<(f64, f64)>, // (x, y) for chart
    pub training_step: f64,
}

impl App {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        Self {
            running: true,
            active_tab: CurrentTab::Dashboard,
            sys,
            chat_input: String::new(),
            chat_history: vec![
                ("User".to_string(), "Hello Llama!".to_string()),
                (
                    "Llama".to_string(),
                    "Greetings human. System online.".to_string(),
                ),
            ],
            training_config: TrainingConfig {
                model_name: "Llama-3-8B".to_string(),
                epochs: "3".to_string(),
                batch_size: "4".to_string(),
                learning_rate: "0.0001".to_string(),
            },
            active_input: 0,
            input_mode: InputMode::Normal,
            is_training: false,
            training_data: vec![],
            training_step: 0.0,
        }
    }

    pub fn on_tick(&mut self) {
        // Refresh stats every tick
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();

        // Simulate training data
        if self.is_training {
            self.training_step += 1.0;
            // Simulated loss curve: decay + noise
            let decay = 5.0 / (1.0 + self.training_step * 0.05);
            let noise = (rand::random::<f64>() - 0.5) * 0.5;
            let loss = (decay + noise).max(0.0);

            self.training_data.push((self.training_step, loss));

            // Keep window reasonable
            if self.training_data.len() > 100 {
                self.training_data.remove(0);
            }
        }
    }

    pub fn next_input(&mut self) {
        self.active_input = (self.active_input + 1) % 4;
    }
}

// --- Main Runner ---

pub fn run() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let tick_rate = Duration::from_millis(100); // Faster tick for smooth charts
    let mut last_tick = std::time::Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.active_tab {
                        CurrentTab::Dashboard | CurrentTab::Chat => {
                            match key.code {
                                KeyCode::Char('q')
                                    if app.active_tab == CurrentTab::Dashboard
                                        || (app.active_tab == CurrentTab::Chat
                                            && app.chat_input.is_empty()) =>
                                {
                                    app.running = false
                                }
                                KeyCode::Tab => {
                                    app.active_tab = match app.active_tab {
                                        CurrentTab::Dashboard => CurrentTab::Chat,
                                        CurrentTab::Chat => CurrentTab::Training,
                                        _ => CurrentTab::Dashboard,
                                    }
                                }
                                // Chat Input handling
                                KeyCode::Char(c) if app.active_tab == CurrentTab::Chat => {
                                    app.chat_input.push(c);
                                }
                                KeyCode::Backspace if app.active_tab == CurrentTab::Chat => {
                                    app.chat_input.pop();
                                }
                                KeyCode::Enter if app.active_tab == CurrentTab::Chat => {
                                    if !app.chat_input.is_empty() {
                                        let msg = app.chat_input.drain(..).collect::<String>();
                                        app.chat_history.push(("User".to_string(), msg));
                                        app.chat_history.push((
                                            "Llama".to_string(),
                                            "Processing...".to_string(),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                        CurrentTab::Training => match app.input_mode {
                            InputMode::Normal => match key.code {
                                KeyCode::Char('q') => app.running = false,
                                KeyCode::Tab => app.active_tab = CurrentTab::Dashboard,
                                KeyCode::Down | KeyCode::Char('j') => app.next_input(),
                                KeyCode::Up | KeyCode::Char('k') => {
                                    if app.active_input > 0 {
                                        app.active_input -= 1;
                                    } else {
                                        app.active_input = 3;
                                    }
                                }
                                KeyCode::Enter => app.input_mode = InputMode::Editing,
                                KeyCode::Char('s') => {
                                    app.is_training = !app.is_training;
                                    if app.is_training {
                                        app.training_data.clear();
                                        app.training_step = 0.0;
                                    }
                                }
                                _ => {}
                            },
                            InputMode::Editing => match key.code {
                                KeyCode::Enter | KeyCode::Esc => app.input_mode = InputMode::Normal,
                                KeyCode::Backspace => {
                                    let val = match app.active_input {
                                        0 => &mut app.training_config.model_name,
                                        1 => &mut app.training_config.epochs,
                                        2 => &mut app.training_config.batch_size,
                                        3 => &mut app.training_config.learning_rate,
                                        _ => unreachable!(),
                                    };
                                    val.pop();
                                }
                                KeyCode::Char(c) => {
                                    let val = match app.active_input {
                                        0 => &mut app.training_config.model_name,
                                        1 => &mut app.training_config.epochs,
                                        2 => &mut app.training_config.batch_size,
                                        3 => &mut app.training_config.learning_rate,
                                        _ => unreachable!(),
                                    };
                                    val.push(c);
                                }
                                _ => {}
                            },
                        },
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = std::time::Instant::now();
        }

        if !app.running {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

// --- UI Rendering ---

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.area());

    // 1. Header (Tabs)
    let titles = vec!["Dashboard", "Chat", "Training [TAB]"];
    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Rusty Monitor ")
                .style(Style::default().fg(Color::Cyan)),
        )
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Yellow),
        )
        .select(match app.active_tab {
            CurrentTab::Dashboard => 0,
            CurrentTab::Chat => 1,
            CurrentTab::Training => 2,
        });
    f.render_widget(tabs, chunks[0]);

    // 2. Main Content
    match app.active_tab {
        CurrentTab::Dashboard => draw_dashboard(f, chunks[1], app),
        CurrentTab::Chat => draw_chat(f, chunks[1], app),
        CurrentTab::Training => draw_training(f, chunks[1], app),
    }

    // 3. Footer
    let status_text = if app.is_training {
        format!(
            "TRAINING IN PROGRESS | Epoch 1/{} | Loss: {:.4}",
            app.training_config.epochs,
            app.training_data.last().unwrap_or(&(0.0, 0.0)).1
        )
    } else {
        format!(
            "Status: Connected | Memory: {:.1} GB / {:.1} GB | GPU: Metal",
            app.sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            app.sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
        )
    };

    let footer_style = if app.is_training {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };

    let footer = Paragraph::new(status_text)
        .style(footer_style)
        .block(Block::default().borders(Borders::TOP));
    f.render_widget(footer, chunks[2]);
}

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // CPU Gauges
    let cpus = app.sys.cpus();
    let _gauge_constraints: Vec<Constraint> = (0..cpus.len())
        .map(|_| Constraint::Ratio(1, cpus.len() as u32))
        .collect();

    let cpu_avg: f32 = if !cpus.is_empty() {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .block(Block::default().title("CPU Average").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Green))
        .ratio((cpu_avg as f64 / 100.0).clamp(0.0, 1.0));

    f.render_widget(gauge, chunks[0]);

    let ram_ratio = app.sys.used_memory() as f64 / app.sys.total_memory() as f64;
    let ram_gauge = Gauge::default()
        .block(Block::default().title("RAM Usage").borders(Borders::ALL))
        .gauge_style(Style::default().fg(Color::Magenta))
        .ratio(ram_ratio.clamp(0.0, 1.0));
    f.render_widget(ram_gauge, chunks[1]);
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // History
    let messages: Vec<ListItem> = app
        .chat_history
        .iter()
        .map(|(sender, msg)| {
            let content = format!("{}: {}", sender, msg);
            let color = if sender == "User" {
                Color::Cyan
            } else {
                Color::Green
            };
            ListItem::new(content).style(Style::default().fg(color))
        })
        .collect();

    let history =
        List::new(messages).block(Block::default().borders(Borders::ALL).title("History"));
    f.render_widget(history, chunks[0]);

    // Input
    let input = Paragraph::new(app.chat_input.as_str())
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[1]);
}

fn draw_training(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
        .split(area);

    // 1. Config Form
    let form_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Model
            Constraint::Length(3), // Epochs
            Constraint::Length(3), // Batch
            Constraint::Length(3), // LR
            Constraint::Min(0),    // Instructions
        ])
        .split(chunks[0]);

    let _input_style = Style::default().fg(Color::Yellow);
    let _active_style = Style::default().fg(Color::Black).bg(Color::Yellow);

    let draw_input = |title: &str, value: &str, index: usize, frame: &mut Frame| {
        let is_selected = app.active_input == index;
        let is_editing = is_selected && app.input_mode == InputMode::Editing;

        let style = if is_selected {
            if is_editing {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Cyan)
            }
        } else {
            Style::default().fg(Color::White)
        };

        let block_title = if is_editing {
            format!("{} (Editing)", title)
        } else {
            title.to_string()
        };

        let input = Paragraph::new(value)
            .style(style)
            .block(Block::default().borders(Borders::ALL).title(block_title));
        frame.render_widget(input, form_chunks[index]);
    };

    draw_input("Model Name", &app.training_config.model_name, 0, f);
    draw_input("Epochs", &app.training_config.epochs, 1, f);
    draw_input("Batch Size", &app.training_config.batch_size, 2, f);
    draw_input("Learning Rate", &app.training_config.learning_rate, 3, f);

    let instructions = Paragraph::new("Controls:\n[Tab]: Switch Tabs\n[Up/Down]: Select Field\n[Enter]: Edit Field\n[s]: Start/Stop Training")
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::ALL).title("Help"));
    f.render_widget(instructions, form_chunks[4]);

    // 2. Training Chart
    let datasets = vec![Dataset::default()
        .name("Loss")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Red))
        .data(&app.training_data)];

    let x_max = app
        .training_data
        .last()
        .map(|d| d.0)
        .unwrap_or(100.0)
        .max(100.0);
    let y_max = app
        .training_data
        .iter()
        .map(|d| d.1)
        .fold(0.0 / 0.0, f64::max)
        .max(5.0); // handle NaN

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title("Training Loss")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Step")
                .bounds([0.0, x_max])
                .labels(vec![Span::raw("0"), Span::raw(format!("{:.0}", x_max))]),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .bounds([0.0, y_max])
                .labels(vec![Span::raw("0.0"), Span::raw(format!("{:.1}", y_max))]),
        );
    f.render_widget(chart, chunks[1]);
}
