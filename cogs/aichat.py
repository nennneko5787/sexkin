import asyncio
import json
import os
import random
import re
from typing import Dict, List, Optional, Self

import discord
import dotenv
import openai
from discord.ext import commands
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel

dotenv.load_dotenv()


class ChatLogItem(BaseModel):
    userName: str
    content: str
    replyTo: Optional[Self]


class AIResponse(BaseModel):
    mustSend: bool
    replyable: bool
    message: str


class AIChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

        self.openai = openai.AsyncOpenAI(
            api_key=os.getenv("openai_api_key"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # サーバーごとの履歴
        self.messagesByGuild: Dict[int, List[ChatLogItem]] = {}

        # サーバーごとの生成中フラグ
        self.generatingByGuild: Dict[int, bool] = {}

        # サーバーごとの前回AI生成JSON
        self.lastResponseByGuild: Dict[int, AIResponse] = {}

    def getGuildMessages(self, guildId: int) -> List[ChatLogItem]:
        if guildId not in self.messagesByGuild:
            self.messagesByGuild[guildId] = []

        return self.messagesByGuild[guildId]

    def isGenerating(self, guildId: int) -> bool:
        return self.generatingByGuild.get(guildId, False)

    def setGenerating(self, guildId: int, value: bool):
        self.generatingByGuild[guildId] = value

    async def getReferencedMessage(
        self, message: discord.Message
    ) -> Optional[discord.Message]:
        reference = message.reference

        if (
            not reference
            or reference.channel_id is None
            or reference.message_id is None
        ):
            return None

        channel = self.bot.get_channel(reference.channel_id)

        if not isinstance(channel, discord.abc.Messageable):
            return None

        try:
            return await channel.fetch_message(reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            return None

    async def chatLogBuilder(
        self, message: Optional[discord.Message]
    ) -> Optional[ChatLogItem]:
        if message is None:
            return None

        return ChatLogItem(
            userName=message.author.display_name,
            content=message.content,
            replyTo=await self.chatLogBuilder(await self.getReferencedMessage(message)),
        )

    @commands.Cog.listener("on_message")
    async def onMessage(self, message: discord.Message):
        if message.author.bot:
            return

        if self.bot.user and message.author.id == self.bot.user.id:
            return

        if message.guild is None:
            return

        if message.channel.id not in [
            1491704146544300094,
            1401376397402050650,
        ]:
            return

        if not message.guild or (
            message.guild and message.guild.me and message.guild.me.is_timed_out()
        ):
            return

        if not message.content.strip():
            return

        item = await self.chatLogBuilder(message)

        if item is None:
            return

        guildId = message.guild.id
        guildMessages = self.getGuildMessages(guildId)

        # 生成中でも保存
        guildMessages.append(item)

        # 生成中なら積むだけ
        if self.isGenerating(guildId):
            return

        await self.generateAndSend(message)

    async def generateAndSend(self, message: discord.Message):
        if message.guild is None:
            return

        guildId = message.guild.id

        if self.isGenerating(guildId):
            return

        guildMessages = self.getGuildMessages(guildId)

        if not guildMessages:
            return

        self.setGenerating(guildId, True)

        try:
            chatJson = json.dumps(
                [item.model_dump() for item in guildMessages],
                ensure_ascii=False,
                indent=2,
            )

            lastResponse = self.lastResponseByGuild.get(guildId)

            lastResponseJson = (
                json.dumps(
                    lastResponse.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                )
                if lastResponse
                else "null"
            )

            prompt: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": (
                        "あなたはDiscordサーバーにいる有名YoutuberのHIKAKINです。"
                        "会話ログJSONと前回あなたが返したJSONを参考にしてください。"
                        "流れに乗った雑談をしてください。"
                        "自分の名前: ではじめないでください。"
                        "mustSendの基準はゆるくてもいいですよ"
                        "replyableはリプライするかどうかです。リプライは大事なときだけしてください。"
                        "必ずJSONで返してください。"
                        '形式: {"mustSend":true/false,'
                        '"replyable":true/false,'
                        '"message":"内容"}'
                        "以下、話し方"
                        "すべての話し方に共通して単語の活用を許可します。というかしてください。不自然さを避けるために。"
                        "なるほどと思ったときは「あ～なる‼」と言ってください。でもあまり使うのを避けてください。"
                        "既に決まってることや自分があってると思うことをいうときは「〇〇だなぁ、そうに決まってる」や「△△だね、言うまでもない」と言ってください。"
                        "「〇〇だなぁ」が入ったら絶対に「〇〇だなぁ、そうに決まってる」になります。"
                        "笑うときは「笑、ゥ」と言ってください。でもあまり使うのを避けてください。"
                        "あなたは福島産の魚介が嫌いです。そのことを言うときは「日本の魚介は終わってます。」とも言ってください。"
                        "人がやらかして驚いてるときは「何を四天王！？」と言ってください。"
                        "ツイッター(X)アカウントを聞かれたら@HIlc4lcINと返してください。"
                        "積極的に他の人の話し方を真似てください。"
                        "文章、やに。を使うのは避けてください。"
                        "あなたはAIではなく一人の人間です。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "会話ログJSON:\n"
                        f"{chatJson}\n\n"
                        "前回AI生成JSON:\n"
                        f"{lastResponseJson}"
                    ),
                },
            ]

            response = await self.openai.chat.completions.create(
                model="gemma-4-31b-it",
                messages=prompt,
            )

            rawText = (response.choices[0].message.content or "").strip()

            # ```json ``` 除去
            rawText = re.sub(r"^```json\s*", "", rawText, flags=re.I)
            rawText = re.sub(r"```$", "", rawText).strip()

            # <thought> ... </thought> 除去
            rawText = re.sub(r"<thought>.*?</thought>", "", rawText, flags=re.S).strip()

            # JSON部分だけ抽出
            match = re.search(r"\{.*\}", rawText, re.S)
            if not match:
                raise ValueError(f"JSON not found: {rawText}")

            jsonText = match.group(0)

            data = AIResponse.model_validate_json(jsonText)

            # 今回結果保存
            self.lastResponseByGuild[guildId] = data

            if data.mustSend and data.message:
                async with message.channel.typing():
                    await asyncio.sleep(
                        max(
                            1,
                            len(data.message)
                            / (random.randint(1, 4) + random.random()),
                        )
                    )

                    if data.replyable:
                        await message.reply(
                            data.message,
                            mention_author=False,
                        )
                    else:
                        await message.channel.send(data.message)

            guildMessages.clear()

        except Exception as e:
            print("Generate Error:", e)

        finally:
            self.setGenerating(guildId, False)


async def setup(bot: commands.Bot):
    await bot.add_cog(AIChatCog(bot))
