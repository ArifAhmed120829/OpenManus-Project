from openai import AsyncOpenAI
import logging
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class Manus:
    def __init__(self, llm_config=None):
        default_config = {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "tinyllama",
            "temperature": 0.3,
            "max_tokens": 4096
        }
        self.config = {**default_config, **(llm_config or {})}
        self.client = AsyncOpenAI(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"]
        )

    async def fetch_url_content(self, url):
        """Fetch HTML content of a URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.text
        except Exception as e:
            logger.error(f"Error fetching URL: {e}")
            return None

    async def analyze_seo(self, html_content):
        """Analyze HTML content for SEO factors"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Basic SEO checks
        title = soup.find('title').text if soup.find('title') else "Missing"
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        h1_tags = [h1.text for h1 in soup.find_all('h1')]
        images_without_alt = [img['src'] for img in soup.find_all('img') if not img.get('alt')]

        return {
            'title': title,
            'meta_description': meta_desc.get('content') if meta_desc else "Missing",
            'h1_count': len(h1_tags),
            'images_missing_alt': len(images_without_alt),
            'basic_issues': {
                'missing_title': title == "Missing",
                'missing_meta_description': meta_desc is None,
                'multiple_h1': len(h1_tags) > 1,
                'images_without_alt': images_without_alt
            }
        }

    async def generate_report(self, url, analysis):
        """Generate markdown report using LLM"""
        prompt = f"""
        Generate a professional SEO audit report for {url} in markdown format.
        Include the following sections:
        1. Basic Information
        2. On-Page SEO Analysis
        3. Technical SEO Issues
        4. Recommendations

        Here's the analysis data:
        {analysis}
        """

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )

        return response.choices[0].message.content

    async def run(self, prompt):
        try:
            # Extract URL from prompt
            if 'url:' in prompt.lower():
                url = prompt.split('url:')[1].split('"')[1]
            else:
                url = prompt.split()[-1].strip('"')

            # Fetch and analyze content
            html = await self.fetch_url_content(url)
            if not html:
                raise ValueError("Failed to fetch URL content")

            analysis = await self.analyze_seo(html)
            report = await self.generate_report(url, analysis)

            # Save report
            with open('report.md', 'w') as f:
                f.write(report)

            print(f"SEO audit report generated and saved to report.md")

        except Exception as e:
            logger.error(f"Error during SEO audit: {e}")
            raise

    async def cleanup(self):
        await self.client.close()
        logger.info("Client connection closed")
