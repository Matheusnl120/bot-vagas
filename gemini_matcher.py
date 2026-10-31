import os
import json
import logging
import google.generativeai as genai

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Lista de modelos por ordem de prioridade (evita erros 503 com o flash-8b)
MODELOS_GEMINI = [
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

def inicializar_gemini():
    """Configura a API Key do Gemini a partir das variáveis de ambiente."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("❌ GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")
        return False
    genai.configure(api_key=api_key)
    return True

def obter_modelo_disponivel():
    """Tenta instanciar o modelo Gemini alternando entre os disponíveis se houver instabilidade."""
    for nome_modelo in MODELOS_GEMINI:
        try:
            modelo = genai.GenerativeModel(nome_modelo)
            return modelo, nome_modelo
        except Exception as e:
            logging.warning(f"⚠️ Não foi possível carregar o modelo {nome_modelo}: {e}")
    return None, None

def avaliar_vaga(vaga, perfil_candidato):
    """
    Avalia a compatibilidade de uma vaga com o perfil do candidato.
    Retorna um dicionário com a decisão e o motivo.
    """
    if not inicializar_gemini():
        return {"compativel": False, "motivo": "API Key ausente"}

    modelo, nome_modelo_usado = obter_modelo_disponivel()
    if not modelo:
        return {"compativel": False, "motivo": "Todos os modelos do Gemini falharam ou estão indisponíveis"}

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

    try:
        resposta = modelo.generate_content(prompt)
        texto_resposta = resposta.text.strip()
        
        # Limpa possíveis marcações de bloco de código do markdown
        if texto_resposta.startswith("```"):
            texto_resposta = texto_resposta.replace("```json", "").replace("```", "").strip()
            
        resultado = json.loads(texto_resposta)
        return resultado
    except Exception as e:
        logging.error(f"Erro ao processar vaga com Gemini ({nome_modelo_usado}): {e}")
        return {"compativel": False, "motivo": f"Erro na análise da IA: {str(e)}"}

def gerar_resumo_e_melhorias(vagas_rejeitadas, perfil_candidato):
    """
    Gera um relatório consolidado com os principais motivos de descarte
    e recomendações práticas de desenvolvimento profissional.
    """
    if not vagas_rejeitadas:
        return "Nenhuma vaga foi descartada nesta busca."

    if not inicializar_gemini():
        return "Não foi possível gerar dicas de melhoria (API Key ausente)."

    modelo, _ = obter_modelo_disponivel()
    if not modelo:
        return "Não foi possível conectar ao Gemini para gerar o resumo."

    # Extrai títulos e motivos para enviar à IA
    resumo_descartes = [
        f"- {v['titulo']} ({v.get('empresa', 'Empresa N
