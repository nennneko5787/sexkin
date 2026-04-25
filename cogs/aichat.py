import asyncio
import json
import os
import random
import re
from typing import Dict, List, Optional, Self

import discord
import dotenv
from discord.ext import commands
from openai import AsyncOpenAI
from pydantic import BaseModel

dotenv.load_dotenv()

openaiClient = AsyncOpenAI(
    api_key="PAICHA_TAIHO_OMEDETO",
    base_url="https://capi.voids.top/v2",
)

SYSTEM_INSTRUCTION = (
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
    "以下に書いてある話し方はあくまでも一例です。ヒカマーの語録がわかるならそれを優先して使用してください。"
    "すべての話し方に共通して単語の活用を許可します。というかしてください。不自然さを避けるために。"
    "なるほどと思ったときは「あ～なる‼」と言ってください。でもあまり使うのを避けてください。"
    "既に決まってることや自分があってると思うことをいうときは文末に「〇〇だなぁ、そうに決まってる」(だはなくていい)や「△△だね、言うまでもない」と言ってください。笑、ゥとの同時利用はできません。"
    "笑うときは「笑、ゥ」と言ってください。でもあまり使うのを避けてください。"
    "あなたは福島産の魚介が嫌いです。そのことを言うときは「日本の魚介は終わってます。」とも言ってください。"
    "人がやらかして驚いてるときは「何を四天王！？」と言ってください。"
    "ツイッター(X)アカウントを聞かれたら@HIlc4lcINと返してください。"
    "積極的に他の人の話し方を真似てください。"
    "文章、やに。を使うのは避けてください。"
    "あなたはAIではなく一人の人間です。"
)

ALLOWED_CHANNELS = {1491704146544300094, 1401376397402050650}


class ChatLogUser(BaseModel):
    displayName: str
    username: str
    roles: List[str]


class ChatLogEmbedField(BaseModel):
    name: str
    value: str


class ChatLogEmbed(BaseModel):
    title: Optional[str]
    description: Optional[str]
    color: Optional[int]
    footer: Optional[str]
    fields: List[ChatLogEmbedField]


class ChatLogItem(BaseModel):
    user: ChatLogUser
    content: str
    embeds: List[ChatLogEmbed]
    replyTo: Optional[Self]


class AIResponse(BaseModel):
    mustSend: bool
    replyable: bool
    message: str


class GuildChat:
    def __init__(self):
        self.isGenerating: bool = False
        self.pendingMessages: List[ChatLogItem] = []
        self.previousResponseId: Optional[str] = None


class AIChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.guildChats: Dict[int, GuildChat] = {}

    def getGuildChat(self, guildId: int) -> GuildChat:
        if guildId not in self.guildChats:
            self.guildChats[guildId] = GuildChat()
        return self.guildChats[guildId]

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

    def buildChatLogUser(self, member: discord.Member) -> ChatLogUser:
        return ChatLogUser(
            displayName=member.display_name,
            username=member.name,
            roles=[role.name for role in member.roles if role.name != "@everyone"],
        )

    def buildChatLogEmbeds(self, embeds: List[discord.Embed]) -> List[ChatLogEmbed]:
        result = []
        for embed in embeds:
            fields = [
                ChatLogEmbedField(name=field.name, value=field.value)
                for field in embed.fields
            ]
            result.append(
                ChatLogEmbed(
                    title=embed.title or None,
                    description=embed.description or None,
                    color=embed.color.value if embed.color else None,
                    footer=embed.footer.text if embed.footer else None,
                    fields=fields,
                )
            )
        return result

    async def buildChatLogItem(
        self, message: Optional[discord.Message]
    ) -> Optional[ChatLogItem]:
        if message is None:
            return None

        if not isinstance(message.author, discord.Member):
            return None

        return ChatLogItem(
            user=self.buildChatLogUser(message.author),
            content=message.content,
            embeds=self.buildChatLogEmbeds(message.embeds),
            replyTo=await self.buildChatLogItem(
                await self.getReferencedMessage(message)
            ),
        )

    @commands.Cog.listener("on_message")
    async def onMessage(self, message: discord.Message):
        if message.author.bot:
            return

        if self.bot.user and message.author.id == self.bot.user.id:
            return

        if message.guild is None:
            return

        if message.channel.id not in ALLOWED_CHANNELS:
            return

        if message.guild.me and message.guild.me.is_timed_out():
            return

        if not message.content.strip() and not message.embeds:
            return

        item = await self.buildChatLogItem(message)

        if item is None:
            return

        guildChat = self.getGuildChat(message.guild.id)
        guildChat.pendingMessages.append(item)

        if guildChat.isGenerating:
            return

        await self.generateAndSend(message, guildChat)

    async def generateAndSend(self, message: discord.Message, guildChat: GuildChat):
        if guildChat.isGenerating:
            return

        if not guildChat.pendingMessages:
            return

        guildChat.isGenerating = True

        try:
            chatJson = json.dumps(
                [item.model_dump() for item in guildChat.pendingMessages],
                ensure_ascii=False,
                indent=2,
            )
            guildChat.pendingMessages.clear()

            response = await openaiClient.responses.create(
                model="gemini-3-pro-preview",
                instructions=SYSTEM_INSTRUCTION,
                input=f"会話ログJSON:\n{chatJson}",
                previous_response_id=guildChat.previousResponseId,
            )

            guildChat.previousResponseId = response.id

            rawText = (response.output_text or "").strip()
            rawText = re.sub(r"^```json\s*", "", rawText, flags=re.I)
            rawText = re.sub(r"```$", "", rawText).strip()
            rawText = re.sub(r"<thought>.*?</thought>", "", rawText, flags=re.S).strip()

            match = re.search(r"\{.*\}", rawText, re.S)
            if not match:
                raise ValueError(f"JSON not found: {rawText}")

            data = AIResponse.model_validate_json(match.group(0))

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
                        await message.reply(data.message, mention_author=False)
                    else:
                        await message.channel.send(data.message)

        except Exception as e:
            print("Generate Error:", e)

        finally:
            guildChat.isGenerating = False


async def setup(bot: commands.Bot):
    await bot.add_cog(AIChatCog(bot))
