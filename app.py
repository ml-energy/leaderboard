from __future__ import annotations

import json
import yaml
import requests
import itertools
import contextlib
from dateutil import parser, tz

import numpy as np
import gradio as gr
import pandas as pd
import plotly.io as pio
import plotly.express as px
from pandas.api.types import is_numeric_dtype, is_float_dtype

pio.templates.default = "plotly_white"


class TableManager:
    def __init__(self, data_dir: str) -> None:
        """Load leaderboard data from CSV files in data_dir.

        Inside `data_dir`, there should be:
        - `models.json`: a JSON file containing information about each model.
        - `schema.yaml`: a YAML file containing the schema of the benchmark.
        - `score.csv`: a CSV file containing the NLP evaluation metrics of each model.
        - `*_benchmark.csv`: CSV files containing the system benchmark results.

        Especially, the `*_benchmark.csv` files should be named after the
        parameters used in the benchmark. For example, for the CSV file that
        contains benchmarking results for A100 and the chat-concise task
        (see `schema.yaml`) for possible choices, the file should be named
        `A100_chat-concise_benchmark.csv`.
        """
        # Load and merge CSV files.
        df = self._read_tables(data_dir)

        # Add the #params column.
        models = json.load(open(f"{data_dir}/models.json"))
        df["parameters"] = df["model"].apply(lambda x: models[x]["params"])

        # Make the first column (model) an HTML anchor to the model's website.
        def format_model_link(model_name: str) -> str:
            url = models[model_name]["url"]
            nickname = models[model_name]["nickname"]
            return (
                f'<a style="text-decoration: underline; text-decoration-style: dotted" '
                f'target="_blank" href="{url}">{nickname}</a>'
            )
        df["model"] = df["model"].apply(format_model_link)

        # Sort by our 'energy efficiency' score.
        df = df.sort_values(by="energy", ascending=True)

        # The full table where all the data are.
        self.full_df = df

        # Default view of the table is to only show the first options.
        self.set_filter_get_df()

    def _read_tables(self, data_dir: str) -> pd.DataFrame:
        """Read tables."""
        df_score = pd.read_csv(f"{data_dir}/score.csv")

        with open(f"{data_dir}/schema.yaml") as file:
            self.schema: dict[str, list] = yaml.safe_load(file)

        res_df = pd.DataFrame()

        # Do a cartesian product of all the choices in the schema
        # and try to read the corresponding CSV files.
        for choice in itertools.product(*self.schema.values()):
            filepath = f"{data_dir}/{'_'.join(choice)}_benchmark.csv"
            with contextlib.suppress(FileNotFoundError):
                df = pd.read_csv(filepath)
                for key, val in zip(self.schema.keys(), choice):
                    df.insert(1, key, val)
                res_df = pd.concat([res_df, df])

        if res_df.empty:
            raise ValueError(f"No benchmark CSV files were read from {data_dir=}.")

        df = pd.merge(res_df, df_score, on=["model"]).round(2)

        # Order columns.
        columns = df.columns.to_list()
        cols_to_order = ["model"]
        cols_to_order.extend(self.schema.keys())
        cols_to_order.append("energy")
        columns = cols_to_order + [col for col in columns if col not in cols_to_order]
        df = df[columns]

        # Delete rows with *any* NaN values.
        df = df.dropna()

        return df

    def _format_msg(self, text: str) -> str:
        """Formats into HTML that prints in Monospace font."""
        return f"<pre style='font-family: monospace'>{text}</pre>"

    def add_column(self, column_name: str, formula: str):
        """Create and add a new column with the given formula."""
        # If the user did not provide the name of the new column,
        # generate a unique name for them.
        if not column_name:
            counter = 1
            while (column_name := f"custom{counter}") in self.full_df.columns:
                counter += 1

        # If the user did not provide a formula, return an error message.
        if not formula:
            return self.cur_df, self._format_msg("Please enter a formula.")

        # If there is an equal sign in the formula, `df.eval` will
        # return an entire DataFrame with the new column, instead of
        # just the new column. This is not what we want, so we check
        # for this case and return an error message.
        if "=" in formula:
            return self.cur_df, self._format_msg("Invalid formula: expr cannot contain '='.")

        # The user may want to update an existing column.
        verb = "Updated" if column_name in self.full_df.columns else "Added"

        # Evaluate the formula and catch any error.
        try:
            # Give the users some helper functions that can be used in the formula
            # like "@sum(response_length)". Also wipe out some global variables.
            col = self.full_df.eval(
                formula,
                local_dict={"sum": sum, "len": len, "max": max, "min": min},
                global_dict={"global_tbm": None},
            )
        except Exception as exc:
            return self.cur_df, self._format_msg(f"Invalid formula: {exc}")

        # If the result is a numeric scalar, make it a Series.
        # We may have deleted some models (rows) form the full dataframe when we
        # called dropna, so we need to query the maximum index instead of taking len.
        if isinstance(col, (int, float)):
            col = pd.Series([col] * (self.full_df.index.max() + 1))
        # We only accept numeric columns.
        if not is_numeric_dtype(col):
            return self.cur_df, self._format_msg("Invalid formula: result must be numeric.")
        # Round if it's floating point.
        if is_float_dtype(col):
            col = col.round(2)

        # If the column already exists, update it.
        if column_name in self.full_df.columns:
            self.full_df[column_name] = col
        else:
            self.full_df.insert(len(self.schema) + 1, column_name, col)

        # If adding a column succeeded, `self.cur_df` should also be updated.
        self.cur_df = self.full_df.loc[self.cur_index]
        return self.cur_df, self._format_msg(f"{verb} column '{column_name}'.")

    def get_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown(choices=columns, label="X"),
            gr.Dropdown(choices=columns, label="Y"),
            gr.Dropdown(choices=["None", *columns], label="Z (optional)"),
        ]

    def update_dropdown(self):
        columns = self.full_df.columns.tolist()[1:]
        return [
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=columns),
            gr.Dropdown.update(choices=["None", *columns]),
        ]

    def set_filter_get_df(self, *filters) -> pd.DataFrame:
        """Set the current set of filters and return the filtered DataFrame."""
        # If the filter is empty, we default to the first choice for each key.
        if not filters:
            filters = [choices[:1] for choices in self.schema.values()]

        index = np.full(len(self.full_df), True)
        for setup, choice in zip(self.schema, filters):
            index = index & self.full_df[setup].isin(choice)
        self.cur_df = self.full_df.loc[index]
        self.cur_index = index
        return self.cur_df

    def plot_scatter(self, width, height, x, y, z):
        # The user did not select either x or y.
        if not x or not y:
            return None, width, height, self._format_msg("Please select both X and Y.")

        # Width and height may be an empty string. Then we set them to 600.
        if not width and not height:
            width, height = "600", "600"
        elif not width:
            width = height
        elif not height:
            height = width
        try:
            width, height = int(width), int(height)
        except ValueError:
            return None, width, height, self._format_msg("Width and height should be positive integers.")

        # Strip the <a> tag from model names.
        text = self.cur_df["model"].apply(lambda x: x.split(">")[1].split("<")[0])
        # Hide model names since they clutter the plots, and only show them on hover.
        if z is None or z == "None" or z == "":
            fig = px.scatter(self.cur_df, x=x, y=y, hover_name=text)
        else:
            fig = px.scatter_3d(self.cur_df, x=x, y=y, z=z, hover_name=text)
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")))
        fig.update_layout(width=width, height=height)

        return fig, width, height, ""


# The global instance of the TableManager should only be used when
# initializing components in the Gradio interface. If the global instance
# is mutated while handling user sessions, the change will be reflected
# in every user session. Instead, the instance provided by gr.State should
# be used.
global_tbm = TableManager("data")

# Fetch the latest update date of the leaderboard repository.
resp = requests.get("https://api.github.com/repos/ml-energy/leaderboard/commits/master")
if resp.status_code != 200:
    current_date = "[Failed to fetch]"
    print("Failed to fetch the latest release date of the leaderboard repository.")
    print(resp.json())
else:
    current_datetime = parser.parse(resp.json()["commit"]["author"]["date"])
    current_date = current_datetime.astimezone(tz.gettz("US/Eastern")).strftime("%Y-%m-%d")

# Custom JS.
# XXX: This is a hack to make the model names clickable.
#      Ideally, we should set `datatype` in the constructor of `gr.DataFrame` to
#      `["markdown"] + ["number"] * (len(df.columns) - 1)` and format models names
#      as an HTML <a> tag. However, because we also want to dynamically add new
#      columns to the table and Gradio < 4.0 does not support updating `datatype` with
#      `gr.DataFrame.update` yet, we need to manually walk into the DOM and replace
#      the innerHTML of the model name cells with dynamically interpreted HTML.
#      Desired feature tracked at https://github.com/gradio-app/gradio/issues/3732
dataframe_update_js = f"""
function format_model_link() {{
    // Iterate over the cells of the first column of the leaderboard table.
    for (let index = 1; index <= {len(global_tbm.full_df)}; index++) {{
        // Get the cell.
        var cell = document.querySelector(
            `#tab-leaderboard > div > div > div > table > tbody > tr:nth-child(${{index}}) > td:nth-child(1) > div > span`
        );

        // If nothing was found, it likely means that now the visible table has less rows
        // than the full table. This happens when the user filters the table. In this case,
        // we should just return.
        if (cell == null) break;

        // This check exists to make this function idempotent.
        // Multiple changes to the Dataframe component may invoke this function,
        // multiple times to the same HTML table (e.g., adding and sorting cols).
        // Thus, we check whether we already formatted the model names by seeing
        // whether the child of the cell is a text node. If it is not,
        // it means we already parsed it into HTML, so we should just return.
        if (cell.firstChild.nodeType != 3) break;

        // Decode and interpret the innerHTML of the cell as HTML.
        var decoded_string = new DOMParser().parseFromString(cell.innerHTML, "text/html").documentElement.textContent;
        var temp = document.createElement("template");
        temp.innerHTML = decoded_string;
        var model_anchor = temp.content.firstChild;

        // Replace the innerHTML of the cell with the interpreted HTML.
        cell.replaceChildren(model_anchor);
    }}

    // Return all arguments as is.
    return arguments
}}
"""

# Custom CSS.
css = """
/* Make ML.ENERGY look like a clickable logo. */
.text-logo {
    color: #27cb63 !important;
    text-decoration: none !important;
}

/* Make the submit button the same color as the logo. */
.btn-submit {
    background: #27cb63 !important;
    color: white !important;
    border: 0 !important;
}

/* Center the plotly plot inside its container. */
.plotly > div {
    margin: auto !important;
}

/* Limit the width of the first column to 300 px. */
table td:first-child,
table th:first-child {
    max-width: 300px;
    overflow: auto;
    white-space: nowrap;
}
"""

block = gr.Blocks(css=css)
with block:
    tbm = gr.State(global_tbm)  # type: ignore
    gr.HTML("<h1><a href='https://ml.energy' class='text-logo'>ML.ENERGY</a> Leaderboard</h1>")

    with gr.Tabs():
        # Tab 1: Leaderboard.
        with gr.TabItem("Leaderboard"):
            # Block 1: Checkboxes to select benchmarking parameters.
            with gr.Row():
                with gr.Box():
                    gr.Markdown("### Benchmark results to show")
                    checkboxes = []
                    for key, choices in global_tbm.schema.items():
                        # Specifying `value` makes everything checked by default.
                        checkboxes.append(gr.CheckboxGroup(choices=choices, value=choices[:1], label=key))

            # Block 2: Leaderboard table.
            with gr.Row():
                dataframe = gr.Dataframe(type="pandas", elem_id="tab-leaderboard")
            # Make sure the models have clickable links.
            dataframe.change(None, None, None, _js=dataframe_update_js)
            # Table automatically updates when users check or uncheck any checkbox.
            for checkbox in checkboxes:
                checkbox.change(TableManager.set_filter_get_df, inputs=[tbm, *checkboxes], outputs=dataframe)

            # Block 3: Allow users to add new columns.
            gr.Markdown("---\n### Add custom columns to the table")
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        colname_input = gr.Textbox(lines=1, label="Custom column name")
                        formula_input = gr.Textbox(lines=1, label="Formula (@sum, @len, @max, and @min are supported)")
                with gr.Column(scale=1):
                    with gr.Row():
                        add_col_btn = gr.Button("Add to table (⏎)", elem_classes=["btn-submit"])
                    with gr.Row():
                        clear_input_btn = gr.Button("Clear")
            with gr.Row():
                add_col_message = gr.HTML("")
            gr.Examples(
                examples=[
                    ["power", "energy / latency"],
                    ["token_per_joule", "response_length / energy"],
                    ["verbose", "response_length > @sum(response_length) / @len(response_length)"],
                ],
                inputs=[colname_input, formula_input],
            )
            colname_input.submit(
                TableManager.add_column,
                inputs=[tbm, colname_input, formula_input],
                outputs=[dataframe, add_col_message],
            )
            formula_input.submit(
                TableManager.add_column,
                inputs=[tbm, colname_input, formula_input],
                outputs=[dataframe, add_col_message],
            )
            add_col_btn.click(
                TableManager.add_column,
                inputs=[tbm, colname_input, formula_input],
                outputs=[dataframe, add_col_message],
            )
            clear_input_btn.click(
                lambda: (None, None, None),
                inputs=None,
                outputs=[colname_input, formula_input, add_col_message],
            )

            # Block 4: Allow users to plot 2D and 3D scatter plots.
            gr.Markdown("---\n### Scatter plot (Hover over marker to show model name)")
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        # Initialize the dropdown choices with the global TableManager with just the original columns.
                        axis_dropdowns = global_tbm.get_dropdown()
                with gr.Column(scale=1):
                    with gr.Row():
                        plot_btn = gr.Button("Plot", elem_classes=["btn-submit"])
                    with gr.Row():
                        clear_plot_btn = gr.Button("Clear")
            with gr.Accordion("Plot size (600 x 600 by default)", open=False):
                with gr.Row():
                    plot_width_input = gr.Textbox("600", lines=1, label="Width (px)")
                    plot_height_input = gr.Textbox("600", lines=1, label="Height (px)")
            with gr.Row():
                plot = gr.Plot()
            with gr.Row():
                plot_message = gr.HTML("")
            add_col_btn.click(TableManager.update_dropdown, inputs=tbm, outputs=axis_dropdowns)  # type: ignore
            plot_width_input.submit(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_height_input.submit(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            plot_btn.click(
                TableManager.plot_scatter,
                inputs=[tbm, plot_width_input, plot_height_input, *axis_dropdowns],
                outputs=[plot, plot_width_input, plot_height_input, plot_message],
            )
            clear_plot_btn.click(
                lambda: (None,) * 7,
                None,
                outputs=[*axis_dropdowns, plot, plot_width_input, plot_height_input, plot_message],
            )

            # Block 5: Leaderboard date.
            with gr.Row():
                gr.HTML(f"<h3 style='color: gray'>Last updated: {current_date}</h3>")

        # Tab 2: About page.
        with gr.TabItem("About"):
            # Read in LEADERBOARD.md
            gr.Markdown(open("LEADERBOARD.md").read())

    # Load the table on page load.
    block.load(lambda: global_tbm.set_filter_get_df(), outputs=dataframe)

block.launch()
