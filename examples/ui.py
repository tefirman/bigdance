#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ui.py
@Time    :   2024/02/20
@Author  :   Taylor Firman
@Version :   1.0
@Contact :   tefirman@gmail.com
@Desc    :   UI components for March Madness bracket app
'''

from shiny import ui

def create_round_header(round_name: str) -> ui.tags.h4:
    """Create a header for a tournament round"""
    return ui.h4(round_name, class_="mt-4 mb-3")

def create_region_column(region: str) -> ui.div:
    """Create a div containing a region's bracket with horizontal round layout"""
    return ui.div(
        ui.h3(f"{region} Region", class_="region-title"),
        ui.row(
            ui.column(
                3,  # Each round takes 1/4 of the width
                create_round_header("First Round"),
                ui.output_ui(f"{region.lower()}_bracket_round1"),
                class_="round-column"
            ),
            ui.column(
                3,
                create_round_header("Second Round"),
                ui.output_ui(f"{region.lower()}_bracket_round2"),
                class_="round-column"
            ),
            ui.column(
                3,
                create_round_header("Sweet 16"),
                ui.output_ui(f"{region.lower()}_bracket_round3"),
                class_="round-column"
            ),
            ui.column(
                3,
                create_round_header("Elite Eight"),
                ui.output_ui(f"{region.lower()}_bracket_round4"),
                class_="round-column"
            ),
            class_="region-rounds"
        ),
        class_="region-container"
    )

# Custom CSS for bracket styling
custom_css = """
.region-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
}

.region-title {
    margin-bottom: 20px;
}

.round-column {
    border-right: 1px solid #eee;
    padding: 0 10px;
    min-height: 600px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.round-column:last-child {
    border-right: none;
}

.bracket-region {
    display: flex;
    flex-direction: column;
}

.game-container {
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 5px;
}

/* First round games are close together */
.round-column:nth-child(1) .game-container {
    margin-bottom: 15px;
}

/* Second round games have more space */
.round-column:nth-child(2) .game-container {
    margin-bottom: 60px;
    margin-top: 30px;
}

/* Sweet 16 games have even more space */
.round-column:nth-child(3) .game-container {
    margin-bottom: 140px;
    margin-top: 70px;
}

/* Elite Eight game is centered in its column */
.round-column:nth-child(4) .game-container {
    margin-top: 0;  /* Remove top margin and let flex centering handle it */
}

.game-container .shiny-input-container {
    margin-bottom: 0;
}

.bracket-region {
    background-color: white;
}

.region-rounds {
    margin: 0;
}

/* Make sure radio buttons and labels don't wrap awkwardly */
.game-container .form-check {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
"""

# Create main app UI
app_ui = ui.page_fluid(
    # Add custom CSS
    ui.tags.style(custom_css),
    
    # App header
    ui.div(
        ui.h2("March Madness Bracket Creator", class_="text-center mb-4"),
        class_="container-fluid"
    ),
    
    # Main layout
    ui.page_sidebar(
        # Sidebar panel
        ui.sidebar(
            ui.div(
                ui.h4("Controls"),
                ui.input_select(
                    "num_entries",
                    "Number of Entries:",
                    [10, 20, 50, 100, 200]
                ),
                ui.input_select(
                    "num_sims",
                    "Number of Simulations:",
                    [1000, 5000, 10000]
                ),
                ui.input_action_button(
                    "simulate", 
                    "Simulate My Bracket",
                    class_="btn-primary w-100 mt-3"
                ),
                ui.div(
                    ui.output_text("simulation_results"),
                    class_="mt-4"
                ),
                # ui.div(
                #     ui.output_text("debug_info"),
                #     class_="mt-4 text-muted small"
                # ),
                class_="sidebar-content"
            )
        ),
        
        # Main panel with bracket regions
        ui.div(
            create_region_column("East"),
            create_region_column("West"),
            create_region_column("South"),
            create_region_column("Midwest"),
            class_="regions-container"
        ),
        
        # Final Rounds
        ui.div(
            ui.h3("Final Rounds", class_="text-center mt-4"),
            ui.row(
                ui.column(
                    6,
                    create_round_header("Final Four"),
                    ui.output_ui("final_four_games")
                ),
                ui.column(
                    6,
                    create_round_header("Championship"),
                    ui.output_ui("championship_game")
                ),
                class_="final-rounds"
            ),
            class_="final-rounds-container mt-4"
        ),
        
        # Footer
        ui.div(
            ui.p(
                "Created by Taylor Firman. See ",
                ui.a("GitHub", href="https://github.com/tefirman/bigdance"),
                " for more details.",
                class_="text-center text-muted small"
            ),
            class_="mt-4"
        )
    )
)
