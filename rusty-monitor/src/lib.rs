use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*};
use std::io::{self};
use std::time::Duration;
use sysinfo::{System};

// --- App State ---

#[derive(PartialEq)]
pub enum CurrentTab {
    Dashboard,
    Chat,
    Training,
}

pub struct App {
    pub running: bool,
    pub active_tab: CurrentTab,
    pub sys: System,
    // Chat state
    pub chat_input: String,
    pub chat_history: Vec<(String, String)>, // (User, AI)
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
                ("Llama".to_string(), "Greetings human. System online.".to_string()),
            ],
        }
    }

    pub fn on_tick(&mut self) {
        // Refresh stats every tick
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();
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
    let tick_rate = Duration::from_millis(250);
    let mut last_tick = std::time::Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        // Global binds
                        KeyCode::Char('q') if app.chat_input.is_empty() => app.running = false,
                        KeyCode::Tab => {
                            app.active_tab = match app.active_tab {
                                CurrentTab::Dashboard => CurrentTab::Chat,
                                CurrentTab::Chat => CurrentTab::Training,
                                CurrentTab::Training => CurrentTab::Dashboard,
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
                                // Mock AI response
                                app.chat_history.push(("Llama".to_string(), "Processing...".to_string()));
                            }
                        }
                        _ => {}
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
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
        .split(f.area());

    // 1. Header (Tabs)
    let titles = vec!["Dashboard [TAB]", "Chat", "Training"];
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(" Rusty Monitor ").style(Style::default().fg(Color::Cyan)))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow))
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
        CurrentTab::Training => draw_training(f, chunks[1]),
    }

    // 3. Footer
    let status_text = format!("Status: Connected | Memory: {:.1} GB / {:.1} GB | GPU: Metal", 
        app.sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0, 
        app.sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
    
    let footer = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::TOP));
    f.render_widget(footer, chunks[2]);
}

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // CPU Gauges
    let cpus = app.sys.cpus(); // In 0.30+, this returns &[Cpu] and Cpu has cpu_usage() method directly
    let _gauge_constraints: Vec<Constraint> = (0..cpus.len()).map(|_| Constraint::Ratio(1, cpus.len() as u32)).collect();
    
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
    let messages: Vec<ListItem> = app.chat_history
        .iter()
        .map(|(sender, msg)| {
            let content = format!("{}: {}", sender, msg);
            let color = if sender == "User" { Color::Cyan } else { Color::Green };
            ListItem::new(content).style(Style::default().fg(color))
        })
        .collect();
        
    let history = List::new(messages)
        .block(Block::default().borders(Borders::ALL).title("History"));
    f.render_widget(history, chunks[0]);

    // Input
    let input = Paragraph::new(app.chat_input.as_str())
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[1]);
}

fn draw_training(f: &mut Frame, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title("Training Monitor (Mock)");
    let text = Paragraph::new("Training graphs will appear here...").block(block);
    f.render_widget(text, area);
}
