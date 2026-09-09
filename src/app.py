"""
Gradio Web Interface for Movie Content Safety Classifier.
Provides a user-friendly web UI for the RAG system and AI Agent.
"""

import gradio as gr
import pandas as pd
from typing import Dict, Any

# Import the RAG chain and configuration
from rag_chain import RAGChain
from agent import MovieSafetyAgent
from config import GROQ_API_KEY, IMDB_MOVIES_PATH
from main import load_movies_from_csv, classify_single_movie


class MovieSafetyApp:
    """
    Gradio web interface for the Movie Safety Classifier.
    """

    def __init__(self):
        """Initialize the RAG chain and agent."""
        print("🔄 Initializing Movie Safety Web App...")

        # Check API key
        if not GROQ_API_KEY:
            print("❌ ERROR: GROQ_API_KEY not found in .env file!")
            print("Please get your free API key from: https://console.groq.com")
            raise ValueError("GROQ_API_KEY not set")

        # Initialize the RAG chain
        print("🔄 Loading RAG chain...")
        self.rag = RAGChain()

        # Initialize the agent
        print("🔄 Loading AI Agent...")
        self.agent = MovieSafetyAgent()

        # Load movie database
        self.movies = load_movies_from_csv()
        self.movie_titles = [m['title'] for m in self.movies]

        print("✅ Movie Safety Web App initialized successfully!")

    def classify_movie_ui(self, title: str, overview: str, genres: str, rating: str) -> str:
        """
        UI wrapper for movie classification.
        """
        if not title:
            return "⚠️ Please enter a movie title."

        if not overview:
            overview = f"A movie titled '{title}'."

        try:
            result = classify_single_movie(self.rag, title, overview, genres, rating)
            return result
        except Exception as e:
            return f"❌ Error: {e}"

    def agent_query_ui(self, question: str) -> str:
        """
        UI wrapper for agent queries.
        """
        if not question:
            return "⚠️ Please enter a question."

        try:
            result = self.agent.ask(question)
            return result
        except Exception as e:
            return f"❌ Error: {e}"

    def batch_classify_ui(self, limit: int) -> str:
        """
        UI wrapper for batch classification.
        """
        if limit < 1:
            return "⚠️ Please enter a number greater than 0."

        if not self.movies:
            return "⚠️ No movies found in database."

        try:
            results = []
            for i, movie in enumerate(self.movies[:int(limit)], 1):
                title = movie.get('title', 'Unknown')
                overview = movie.get('overview', '')
                genres = movie.get('genres', 'Unknown')
                rating = movie.get('rating', 'Unknown')

                result = classify_single_movie(self.rag, title, overview, genres, rating)

                # Extract classification and explanation
                classification = "Not safe" if "Not safe" in result else "Safe"

                results.append({
                    'title': title,
                    'classification': classification,
                    'details': result
                })

            # Format results
            output = f"📊 Batch Classification Results ({len(results)} movies)\n"
            output += "=" * 50 + "\n\n"

            for r in results:
                emoji = "✅" if "Safe" in r['classification'] else "❌"
                output += f"{emoji} {r['title']}: {r['classification']}\n"

            output += "\n" + "=" * 50
            output += "\n\n📋 Detailed results:\n\n"

            for r in results:
                output += f"🎬 {r['title']}\n"
                output += f"{r['details']}\n"
                output += "-" * 40 + "\n\n"

            return output
        except Exception as e:
            return f"❌ Error: {e}"

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        """
        with gr.Blocks(title="Movie Safety Classifier") as interface:

            # Header
            gr.Markdown("""
            # 🎬 Movie Content Safety Classifier
            ### AI-powered RAG system that determines if a movie is appropriate for children aged 5-10
            """)

            with gr.Tabs():

                # Tab 1: Single Movie Classification
                with gr.TabItem("🔍 Classify a Movie"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            title_input = gr.Textbox(
                                label="🎬 Movie Title",
                                placeholder="Enter movie title (e.g., The Lion King)",
                                lines=1
                            )
                            overview_input = gr.Textbox(
                                label="📝 Movie Overview",
                                placeholder="Enter a brief description or plot summary...",
                                lines=5
                            )
                            with gr.Row():
                                genres_input = gr.Textbox(
                                    label="🎭 Genres",
                                    placeholder="Action, Drama, Comedy...",
                                    lines=1
                                )
                                rating_input = gr.Textbox(
                                    label="⭐ Rating",
                                    placeholder="7.5",
                                    lines=1
                                )
                            classify_btn = gr.Button("🔍 Classify Movie", variant="primary")

                        with gr.Column(scale=2):
                            output_text = gr.Textbox(
                                label="📌 Result",
                                lines=15,
                                interactive=False
                            )

                    # Examples
                    gr.Markdown("### 📋 Example Movies")
                    gr.Examples(
                        examples=[
                            ["The Lion King", "A young lion prince flees his kingdom after the murder of his father and learns about responsibility and friendship.", "Animation, Adventure, Drama", "8.5"],
                            ["The Dark Knight", "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", "Action, Crime, Drama", "9.0"],
                            ["Finding Nemo", "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home.", "Animation, Adventure, Comedy", "8.2"],
                            ["Pulp Fiction", "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.", "Crime, Drama", "8.9"],
                            ["The Conjuring", "Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse.", "Horror, Mystery, Thriller", "7.5"],
                        ],
                        inputs=[title_input, overview_input, genres_input, rating_input],
                        label="Click an example to load it"
                    )

                    classify_btn.click(
                        fn=self.classify_movie_ui,
                        inputs=[title_input, overview_input, genres_input, rating_input],
                        outputs=output_text
                    )

                # Tab 2: AI Agent Chat
                with gr.TabItem("🤖 AI Agent"):
                    gr.Markdown("""
                    ### Ask the AI Agent complex questions about movie safety
                    
                    **Example questions:**
                    - *"Can you find me a movie like The Lion King that is appropriate for a 5-year-old?"*
                    - *"Is Jurassic Park safe for children?"*
                    - *"What are the best family movies in the database?"*
                    - *"Tell me about The Avengers."*
                    """)

                    with gr.Row():
                        with gr.Column(scale=2):
                            question_input = gr.Textbox(
                                label="💭 Your Question",
                                placeholder="Ask anything about movie safety...",
                                lines=3
                            )
                            agent_btn = gr.Button("🤖 Ask Agent", variant="primary")

                        with gr.Column(scale=3):
                            agent_output = gr.Textbox(
                                label="📌 Response",
                                lines=15,
                                interactive=False
                            )

                    agent_btn.click(
                        fn=self.agent_query_ui,
                        inputs=[question_input],
                        outputs=agent_output
                    )

                # Tab 3: Batch Classification
                with gr.TabItem("📊 Batch Classification"):
                    gr.Markdown("""
                    ### Classify multiple movies at once
                    
                    This will classify the first N movies from your database.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            limit_input = gr.Number(
                                label="📊 Number of Movies",
                                value=5,
                                minimum=1,
                                maximum=30,
                                step=1
                            )
                            batch_btn = gr.Button("📊 Run Batch Classification", variant="primary")

                        with gr.Column(scale=2):
                            batch_output = gr.Textbox(
                                label="📌 Results",
                                lines=20,
                                interactive=False
                            )

                    batch_btn.click(
                        fn=self.batch_classify_ui,
                        inputs=[limit_input],
                        outputs=batch_output
                    )

                # Tab 4: About
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ## 🎬 Movie Content Safety Classifier
                    
                    This application uses **RAG (Retrieval-Augmented Generation)** to determine if movies are appropriate for children aged 5-10.
                    """)

        return interface


def main():
    """Launch the Gradio interface."""
    print("=" * 60)
    print("🎬 Movie Content Safety Classifier - Web Interface")
    print("=" * 60)

    app = MovieSafetyApp()
    interface = app.create_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    main()