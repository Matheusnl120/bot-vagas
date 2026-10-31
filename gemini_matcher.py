import os
import json
import logging
from google import genai

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Lista de modelos por ordem de prioridade (evita erros 503 com versões leves)
MODELOS_GEMINI = [
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

def carregar_perfil(caminho="perfil_candidato.json"):
    """Carrega o perfil do candidato a partir do arquivo JSON (exigido pelo run.py)."""
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logging.warning(f"⚠️ Ficheiro {caminho} não encontrado. A usar perfil vazio.")
            return {}
    except Exception as e:
        logging.error(f"Erro ao carregar perfil: {e}")
        return {}

def criar_cliente_gemini():
    """Inicializa e retorna o cliente oficial da SDK Google GenAI."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("❌ GEMINI_API_KEY não encontrada nas variáveis de ambiente.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logging.error(f"Erro ao inicializar cliente Gemini: {e}")
        return None

def analisar_compatibilidade_vaga(vaga, perfil_candidato):
    """
    Avalia a compatibilidade de uma vaga com o perfil do candidato.
    Nome original preservado para evitar qualquer ImportError no run.py.
    """
    client = criar_cliente_gemini()
    if not client:
        return {"compativel": False, "motivo": "API Key ausente ou inválida"}

    prompt = f"""
    Você é um recrutador técnico experiente. Avalie se a vaga abaixo é compatível com o perfil do candidato.

    --- PERFIL DO CANDIDATO ---
    {json.dumps(perfil_candidato, ensure_ascii=False, indent=2)}

    --- DETALHES DA VAGA ---
    Título: {vaga.get('titulo', 'N/A')}
    Empresa: {vaga.get('empresa', 'N/A')}
    Localização/Modalidade: {vaga.get('localizacao', 'N/A')}
    Descrição/Requisitos: {vaga.get('descricao', 'N/A')}

    --- REGRAS DE AVALIAÇÃO ---
    1. Retorne APENAS um JSON válido no seguinte formato exato (sem formatação markdown ```json):
    {{
        "compativel": true ou false,
        "pontuacao": 0 a 100,
        "motivo": "Justificativa curta e direta em 1 ou 2 frases sobre a decisão."
    }}
    """

    for nome_modelo in MODELOS_GEMINI:
        try:
            response = client.models.generate_content(
                model=nome_modelo,
                contents=prompt,
            )
            texto_resposta = response.text.strip()
            
            if texto_resposta.startswith("```"):
                texto_resposta = texto_resposta.replace("```json", "").replace("```", "").strip()
                
            return json.loads(texto_resposta)
        except Exception as e:
            logging.warning(f"⚠️ Falha no modelo {nome_modelo}: {e}")

    return {"compativel": False, "motivo": "Todos os modelos do Gemini falharam ou estão indisponíveis"}

# Alias de compatibilidade caso outras partes chamem 'avaliar_vaga'
avaliar_vaga = analisar_compatibilidade_vaga

def gerar_resumo_e_melhorias(vagas_rejeitadas, perfil_candidato):
    """
    Gera um relatório consolidado com os principais motivos de descarte
    e recomendações práticas de desenvolvimento profissional para o currículo.
    """
    if not vagas_rejeitadas:
        return "Nenhuma vaga foi descartada nesta busca."

    client = criar_cliente_gemini()
    if not client:
        return "Não foi possível gerar dicas de melhoria (API Key ausente)."

    resumo_descartes = []
    for v in vagas_rejeitadas:
        titulo = v.get('titulo', 'N/A')
        empresa = v.get('empresa', 'Empresa N/A')
        motivo = v.get('motivo', 'Sem motivo especificado')
        resumo_descartes.append(f"- {titulo} ({empresa}): {motivo}")

    prompt = f"""
    Você é um mentor de carreira de TI e Suporte. Analise a lista de vagas que foram REJEITADAS para o candidato {perfil_candidato.get('nome', 'Candidato')}:

    --- VAGAS DESCARTADAS E MOTIVOS ---
    {chr(10).join(resumo_descartes)}

    --- TAREFA ---
    Elabore um pequeno relatório informativo (máximo 4 a 5 linhas/tópicos) formatado para ser enviado via Telegram contendo:
    1. ❌ Principais motivos das exclusões (ex: falta de requisitos específicos, nível de senioridade exigido, modalidade).
    2. 💡 Sugestão prática do que o candidato pode estudar ou melhorar no currículo para aumentar os acertos.
    """

    for nome_modelo in MODELOS_GEMINI:
        try:
            response = client.models.generate_content(
                model=nome_modelo,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logging.warning(f"⚠️ Falha ao gerar resumo no modelo {nome_modelo}: {e}")

    return "Erro ao consolidar feedback de melhorias com o Gemini."
