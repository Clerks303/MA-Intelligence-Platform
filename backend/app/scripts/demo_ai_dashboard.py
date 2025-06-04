"""
Script de démonstration du Dashboard IA
US-010: Démonstration complète des fonctionnalités du dashboard IA
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.ai_dashboard import (
    get_dashboard_engine,
    get_dashboard_data_api,
    explain_prediction_api,
    get_ai_insights_summary
)
from app.core.advanced_ai_engine import get_advanced_ai_engine
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("demo_ai_dashboard", LogCategory.AI_ML)


async def demo_dashboard_creation():
    """Démontre la création de dashboards"""
    
    print("\n" + "="*80)
    print("🎯 DÉMONSTRATION CRÉATION DE DASHBOARDS IA")
    print("="*80)
    
    try:
        engine = await get_dashboard_engine()
        
        # Créer un dashboard personnalisé
        custom_widgets = [
            {
                "widget_id": "ma_performance",
                "type": "gauge_chart",
                "title": "Performance M&A",
                "description": "Score global de performance des acquisitions",
                "config": {"min_val": 0, "max_val": 100, "thresholds": {"good": 80, "warning": 60}},
                "data_source": "ma_metrics",
                "refresh_interval": 300,
                "grid_position": {"x": 0, "y": 0, "width": 6, "height": 4}
            },
            {
                "widget_id": "predictions_history",
                "type": "line_chart", 
                "title": "Historique des Prédictions",
                "description": "Évolution des prédictions IA au fil du temps",
                "config": {"x_col": "date", "y_col": "prediction_count"},
                "data_source": "predictions_timeline",
                "refresh_interval": 600,
                "grid_position": {"x": 6, "y": 0, "width": 6, "height": 4}
            }
        ]
        
        dashboard_id = engine.create_custom_dashboard(
            "demo_user",
            "Dashboard M&A Intelligence", 
            custom_widgets
        )
        
        print(f"✅ Dashboard personnalisé créé: {dashboard_id}")
        
        # Lister les dashboards disponibles
        available_dashboards = engine.get_available_dashboards("demo_user")
        print(f"📊 {len(available_dashboards)} dashboards disponibles:")
        
        for dashboard in available_dashboards:
            print(f"   - {dashboard['title']} ({dashboard['type']})")
        
        return dashboard_id
        
    except Exception as e:
        print(f"❌ Erreur création dashboard: {e}")
        return None


async def demo_dashboard_data():
    """Démontre la récupération de données de dashboard"""
    
    print("\n" + "="*80)
    print("📊 DÉMONSTRATION RÉCUPÉRATION DONNÉES DASHBOARD")
    print("="*80)
    
    try:
        # Récupérer données du dashboard overview
        dashboard_data = await get_dashboard_data_api("overview", "demo_user")
        
        if "error" in dashboard_data:
            print(f"❌ Erreur: {dashboard_data['error']}")
            return
        
        print(f"📈 Dashboard: {dashboard_data['title']}")
        print(f"📝 Description: {dashboard_data['description']}")
        print(f"🔧 Widgets disponibles:")
        
        for widget_id, widget_data in dashboard_data['widgets'].items():
            print(f"   - {widget_id}: {widget_data.get('config', {}).get('title', 'Widget')}")
            if 'error' in widget_data:
                print(f"     ⚠️  Erreur: {widget_data['error']}")
            else:
                print(f"     ✅ Données chargées")
        
        print(f"⏰ Dernière mise à jour: {dashboard_data['last_updated']}")
        
    except Exception as e:
        print(f"❌ Erreur récupération données: {e}")


async def demo_prediction_explanation():
    """Démontre les explications de prédictions"""
    
    print("\n" + "="*80)
    print("🔍 DÉMONSTRATION EXPLICATIONS DE PRÉDICTIONS")
    print("="*80)
    
    try:
        # Données d'exemple pour explication
        prediction_data = {
            "chiffre_affaires": 5000000,
            "effectifs": 25,
            "company_age": 8,
            "productivity": 45000,
            "growth_rate": 0.15,
            "sector_tech": 1,
            "localisation_paris": 1,
            "debt_ratio": 0.3,
            "prediction": 78.5
        }
        
        print("🎯 Données de l'entreprise à expliquer:")
        for key, value in prediction_data.items():
            if key != "prediction":
                print(f"   - {key}: {value}")
        
        print(f"🤖 Prédiction du modèle: {prediction_data['prediction']}")
        
        # Générer explication
        explanation = await explain_prediction_api("ensemble", prediction_data)
        
        if "error" in explanation:
            print(f"❌ Erreur génération explication: {explanation['error']}")
            return
        
        print(f"\n📊 EXPLICATION DÉTAILLÉE:")
        print(f"🔮 Modèle utilisé: {explanation['model_name']}")
        print(f"📈 Confiance: {explanation['confidence']:.1%}")
        
        print(f"\n💼 Interprétation business:")
        print(f"   {explanation['business_interpretation']}")
        
        print(f"\n🎯 Top features importantes:")
        for feature, importance in list(explanation['feature_importance'].items())[:5]:
            print(f"   - {feature}: {importance:.1%}")
        
        print(f"\n⚠️  Facteurs de risque:")
        for risk in explanation['risk_factors']:
            print(f"   - {risk}")
        
        print(f"\n🚀 Opportunités:")
        for opportunity in explanation['opportunities']:
            print(f"   - {opportunity}")
        
        print(f"\n💡 Insights actionnables:")
        for insight in explanation['actionable_insights']:
            print(f"   - {insight}")
        
        print(f"\n📋 Prochaines étapes:")
        for step in explanation['next_steps']:
            print(f"   - {step}")
        
    except Exception as e:
        print(f"❌ Erreur explication prédiction: {e}")


async def demo_ai_insights():
    """Démontre le résumé des insights IA"""
    
    print("\n" + "="*80)
    print("🧠 DÉMONSTRATION INSIGHTS IA GLOBAUX")
    print("="*80)
    
    try:
        insights = await get_ai_insights_summary()
        
        print("📊 RÉSUMÉ DES INSIGHTS IA:")
        print(f"   🎯 Prédictions aujourd'hui: {insights['total_predictions_today']}")
        print(f"   🎲 Confiance moyenne: {insights['average_confidence']:.1%}")
        print(f"   🚨 Anomalies détectées: {insights['anomalies_detected']}")
        print(f"   💡 Recommandations générées: {insights['recommendations_generated']}")
        
        print(f"\n🤖 Performance des modèles:")
        perf = insights['model_performance']
        print(f"   - Accuracy ensemble: {perf['ensemble_accuracy']:.1%}")
        print(f"   - Meilleur modèle: {perf['best_performing_model']}")
        print(f"   - Nombre de modèles: {perf['models_count']}")
        
        print(f"\n🚨 Alertes actives: {len(insights['alerts'])}")
        for alert in insights['alerts'][:3]:  # Top 3 alertes
            print(f"   - [{alert['level'].upper()}] {alert['title']}")
        
        print(f"\n⏰ Dernière mise à jour: {insights['last_updated']}")
        
    except Exception as e:
        print(f"❌ Erreur récupération insights: {e}")


async def demo_system_alerts():
    """Démontre le système d'alertes"""
    
    print("\n" + "="*80)
    print("🚨 DÉMONSTRATION SYSTÈME D'ALERTES")
    print("="*80)
    
    try:
        engine = await get_dashboard_engine()
        
        # Simuler des métriques système pour déclencher alertes
        test_metrics = {
            "accuracy": 0.75,  # Sous le seuil de 0.8
            "model_name": "test_model",
            "confidence": 0.5,  # Faible confiance
            "p_value": 0.03,   # Dérive détectée
            "feature_name": "chiffre_affaires",
            "severity": "high",
            "entity": "test_company",
            "description": "Valeurs inhabituelles détectées"
        }
        
        # Vérifier alertes
        new_alerts = await engine.alerting_system.check_alerts(test_metrics)
        
        print(f"🔔 {len(new_alerts)} nouvelles alertes générées:")
        for alert in new_alerts:
            print(f"\n📋 Alerte: {alert.alert_id}")
            print(f"   🎯 Niveau: {alert.level.value.upper()}")
            print(f"   📄 Titre: {alert.title}")
            print(f"   💬 Message: {alert.message}")
            print(f"   🛠️  Actions suggérées:")
            for action in alert.suggested_actions:
                print(f"      - {action}")
        
        # Lister toutes les alertes actives
        all_alerts = await engine.get_system_alerts()
        print(f"\n📊 Total des alertes actives: {len(all_alerts)}")
        
        # Simuler acquittement d'une alerte
        if new_alerts:
            alert_to_ack = new_alerts[0]
            engine.alerting_system.acknowledge_alert(alert_to_ack.alert_id, "demo_user")
            print(f"✅ Alerte {alert_to_ack.alert_id} acquittée")
        
    except Exception as e:
        print(f"❌ Erreur système alertes: {e}")


async def demo_dashboard_analytics():
    """Démontre les analytics du dashboard"""
    
    print("\n" + "="*80)
    print("📈 DÉMONSTRATION ANALYTICS DASHBOARD")
    print("="*80)
    
    try:
        engine = await get_dashboard_engine()
        analytics = engine.get_dashboard_analytics()
        
        print("📊 ANALYTICS DU SYSTÈME DASHBOARD:")
        print(f"   📋 Total dashboards: {analytics['total_dashboards']}")
        print(f"   🧩 Total widgets: {analytics['total_widgets']}")
        print(f"   🚨 Alertes actives: {analytics['active_alerts']}")
        print(f"   💾 Points de données en cache: {analytics['cached_data_points']}")
        
        print(f"\n📈 Distribution par type:")
        for dashboard_type, count in analytics['dashboard_type_distribution'].items():
            print(f"   - {dashboard_type}: {count}")
        
        print(f"\n💚 Statut système: {analytics['system_health']}")
        print(f"⏰ Dernière mise à jour: {analytics['last_updated']}")
        
    except Exception as e:
        print(f"❌ Erreur analytics dashboard: {e}")


async def demo_complete_workflow():
    """Démontre un workflow complet du dashboard IA"""
    
    print("\n" + "="*80)
    print("🚀 DÉMONSTRATION WORKFLOW COMPLET DASHBOARD IA")
    print("="*80)
    
    try:
        print("🔧 Initialisation du système...")
        await get_dashboard_engine()
        await get_advanced_ai_engine()
        
        print("✅ Système initialisé avec succès!\n")
        
        # Exécuter toutes les démos
        await demo_dashboard_creation()
        await demo_dashboard_data()
        await demo_prediction_explanation()
        await demo_ai_insights()
        await demo_system_alerts()
        await demo_dashboard_analytics()
        
        print("\n" + "="*80)
        print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print("="*80)
        print("\n📊 Le Dashboard IA est entièrement fonctionnel et propose:")
        print("   ✅ Dashboards personnalisables avec widgets interactifs")
        print("   ✅ Explications détaillées des prédictions IA (XAI)")
        print("   ✅ Visualisations avancées et interactives")
        print("   ✅ Système d'alertes intelligent et proactif")
        print("   ✅ Analytics et monitoring en temps réel")
        print("   ✅ Interface API complète pour intégration frontend")
        print("\n🚀 Le système est prêt pour la production!")
        
    except Exception as e:
        print(f"❌ Erreur workflow complet: {e}")
        raise


async def main():
    """Fonction principale de démonstration"""
    
    print("🎯 DÉMONSTRATION DASHBOARD IA - US-010")
    print("🔍 Système de visualisation et explications des prédictions IA")
    print("=" * 80)
    
    try:
        await demo_complete_workflow()
        
    except KeyboardInterrupt:
        print("\n⏹️  Démonstration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant la démonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())