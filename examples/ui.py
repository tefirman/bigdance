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

def create_region_column(region: str) -> ui.column:
    """Create a column containing a region's bracket"""
    return ui.column(
        6,
        ui.h3(f"{region} Region"),
        ui.div(
            create_round_header("First Round"),
            ui.output_ui(f"{region.lower()}_bracket_round1"),
            create_round_header("Second Round"),
            ui.output_ui(f"{region.lower()}_bracket_round2"),
            create_round_header("Sweet 16"),
            ui.output_ui(f"{region.lower()}_bracket_round3"),
            create_round_header("Elite Eight"),
            ui.output_ui(f"{region.lower()}_bracket_round4"),
            class_="region-container"
        )
    )

# Custom CSS for bracket styling
custom_css = """
.region-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
}

.game-container {
    margin-bottom: 15px;
    padding: 10px;
    background-color: #f8f9fa;
    border-radius: 5px;
}

.game-container .shiny-input-container {
    margin-bottom: 0;
}

.bracket-region {
    background-color: white;
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
                    "conference",
                    "Filter by Conference:",
                    ["All Games", "ACC", "Big 12", "Big East", "Big Ten", "Pac-12", "SEC"]
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
                ui.div(
                    ui.output_text("debug_info"),
                    class_="mt-4 text-muted small"
                ),
                class_="sidebar-content"
            )
        ),
        
        # Main panel with bracket regions
        ui.row(
            create_region_column("East"),
            create_region_column("West"),
        ),
        ui.row(
            create_region_column("South"),
            create_region_column("Midwest"),
        ),
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
                )
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
